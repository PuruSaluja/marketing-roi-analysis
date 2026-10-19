import copy
import dataclasses
import itertools
import json

import numpy as np
import pytest

from marketing_roi import holdout, identify, mmm, optimise
from marketing_roi.generate import GeneratorConfig, generate
from marketing_roi.saturation import ResponseCurve, hill


def log_curve(beta: float, scale: float) -> ResponseCurve:
    return ResponseCurve("log", beta, scale=scale)


def s_curve(beta: float, k: float, shape: float) -> ResponseCurve:
    return ResponseCurve("hill", beta, half_saturation=k, shape=shape)


# --- allocate --------------------------------------------------------------------------------


def test_concave_curves_get_the_split_that_equalises_marginal_returns() -> None:
    # beta / (scale + x) = lambda on every channel: x = beta / lambda - scale.
    curves = {"a": log_curve(900, 300), "b": log_curve(500, 100), "c": log_curve(300, 200)}
    total = 2000.0
    inv = (total + 600) / 1700
    expected = {"a": 900 * inv - 300, "b": 500 * inv - 100, "c": 300 * inv - 200}
    got = optimise.allocate(curves, total)
    assert sum(got.spend.values()) == pytest.approx(total, rel=1e-12)
    for name, x in expected.items():
        assert got.spend[name] == pytest.approx(x, abs=got.resolution)
    marginals = list(got.marginal.values())
    assert max(marginals) - min(marginals) < 0.001 * min(marginals)
    assert got.resolution == pytest.approx(0.5)
    assert got.revenue == pytest.approx(sum(curves[n].response(got.spend[n]) for n in curves))


def test_the_grid_optimum_is_the_best_of_every_split_on_the_grid() -> None:
    curves = {"a": s_curve(500, 30, 3.0), "b": s_curve(300, 10, 1.5), "c": s_curve(400, 25, 2.2)}
    lower, upper = {"a": 6, "b": 0, "c": 3}, {"a": 50, "b": 40, "c": 60}
    got = optimise.allocate(curves, 60, lower, upper, steps=60)
    best = max(
        sum(curves[n].response(float(x)) for n, x in zip(curves, split, strict=True))
        for split in itertools.product(range(61), repeat=3)
        if sum(split) == 60
        and all(lower[n] <= x <= upper[n] for n, x in zip(curves, split, strict=True))
    )
    assert got.revenue == pytest.approx(best, rel=1e-12)
    assert all(lower[n] <= got.spend[n] <= upper[n] for n in curves)


def test_two_s_curves_on_a_small_budget_are_not_split_evenly() -> None:
    # An even split equalises the marginal returns, but below the S's inflection (79
    # here) each curve is convex there, so the even split is a minimum, not a maximum.
    curves = {"a": s_curve(1000, 100, 3.0), "b": s_curve(1000, 100, 3.0)}
    got = optimise.allocate(curves, 120)
    even = 2 * float(curves["a"].response(60.0))
    assert got.revenue > 1.5 * even
    assert sorted(got.spend.values()) == pytest.approx([0.0, 120.0])


def test_a_bound_holds_the_best_channel_back() -> None:
    curves = {"a": log_curve(5000, 100), "b": log_curve(100, 100)}
    got = optimise.allocate(curves, 1000, upper={"a": 600})
    assert got.spend == pytest.approx({"a": 600.0, "b": 400.0})
    assert got.marginal["a"] > got.marginal["b"]  # still worth more, but capped


def test_bounds_bind_on_every_channel_and_may_pin_one() -> None:
    curves = {"a": log_curve(10, 100), "b": log_curve(5000, 100), "c": log_curve(4000, 100)}
    got = optimise.allocate(curves, 1000, {"a": 300.0}, {"b": 350.0})
    # a is worth least but cannot go below 300; b is capped, so c takes the rest.
    assert got.spend == pytest.approx({"a": 300.0, "b": 350.0, "c": 350.0})
    assert got.marginal["b"] > got.marginal["c"]
    pinned = optimise.allocate(curves, 1000, {"b": 400.0}, {"b": 400.0})
    assert pinned.spend["b"] == pytest.approx(400.0)


def test_identical_curves_share_the_budget_evenly_and_one_channel_takes_it_all() -> None:
    curves = {"a": log_curve(500, 100), "b": log_curve(500, 100)}
    assert optimise.allocate(curves, 400).spend == pytest.approx({"a": 200.0, "b": 200.0})
    assert optimise.allocate({"a": log_curve(500, 100)}, 400).spend == {"a": 400.0}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"total": 0.0}, "above 0"),
        ({"total": -5.0}, "total"),
        ({"total": float("nan")}, "total"),
        ({"total": True}, "total"),
        ({"steps": 0}, "steps"),
        ({"steps": 2.5}, "steps"),
        ({"steps": True}, "steps"),
        ({"lower": {"z": 1.0}}, "'z'"),
        ({"lower": {"a": 50.0}, "upper": {"a": 40.0}}, "below its lower"),
        ({"lower": {"a": 60.0, "b": 60.0}}, "cannot add up"),
        ({"upper": {"a": 10.0, "b": 10.0}}, "cannot add up"),
        ({"lower": {"a": 10.3}, "upper": {"a": 10.6}}, "no point of the grid"),
        ({"upper": {"a": float("inf")}}, "upper bound for a"),
    ],
)
def test_bad_allocations_are_refused(kwargs, match) -> None:
    curves = {"a": log_curve(500, 100), "b": log_curve(500, 100)}
    args = {"total": 100.0, "steps": 100} | kwargs
    with pytest.raises(ValueError, match=match):
        optimise.allocate(curves, args.pop("total"), **args)


def test_no_channels_is_refused() -> None:
    with pytest.raises(ValueError, match="no channels"):
        optimise.allocate({}, 100.0)


# --- a planned reallocation on a generated run ----------------------------------------------


@pytest.fixture(scope="module")
def planned():
    return optimise.plan(5, fits=("committed", "lift"), days=400, steps=500)


def test_every_fit_spends_the_current_total_within_the_bounds(planned) -> None:
    current, total = planned["current"], planned["total"]
    assert total == pytest.approx(sum(current.values()))
    assert planned["train_days"] == 309 and planned["resolution"] == pytest.approx(total / 500)
    for f in planned["fits"].values():
        assert sum(f["spend"].values()) == pytest.approx(total, rel=1e-12)
        for c, x in f["spend"].items():
            assert 0.5 * current[c] - 1e-9 <= x <= 2.0 * current[c] + 1e-9
        assert f["projected_gain"] >= -1e-9  # the current split is a feasible one


def test_the_true_gain_is_scored_on_the_true_revenue_curves(planned) -> None:
    data = generate(GeneratorConfig(seed=5, days=400))
    cd = data.channel_daily
    spend = np.zeros((4, 400))
    spend[cd["channel"], cd["day"]] = cd["spend_cents"] / 100
    current = {c.name: spend[i, :309].mean() for i, c in enumerate(data.config.channels)}
    assert planned["current"] == pytest.approx(current)

    def true_revenue(alloc) -> float:
        out = 0.0
        for ch in data.config.channels:
            k = ch.half_saturation * ch.daily_spend
            out += data.betas[ch.name] * 90.0 * hill(alloc[ch.name], k, ch.hill_shape)
        return out

    for f in planned["fits"].values():
        assert f["true_gain"] == pytest.approx(true_revenue(f["spend"]) - true_revenue(current))
        # No split on the grid does better on the truth than the true curves' optimum.
        assert f["true_gain"] <= planned["true_optimum"]["true_gain"] + 1e-9
    assert planned["true_optimum"]["true_gain"] > 0
    revenue = mmm.inputs_from_dataset(data).new_revenue[:309].mean()
    assert planned["new_customer_revenue"] == pytest.approx(revenue)
    for f in planned["fits"].values():
        assert f["true_gain_share"] == pytest.approx(f["true_gain"] / revenue)


def test_the_spread_counts_fits_moving_each_channel(planned) -> None:
    s = planned["spread"]
    tol = planned["resolution"]
    for c, v in s["channels"].items():
        values = [f["spend"][c] for f in planned["fits"].values()]
        assert (v["min"], v["max"]) == (min(values), max(values))
        assert v["up"] == sum(x > planned["current"][c] + tol for x in values)
        assert v["down"] == sum(x < planned["current"][c] - tol for x in values)
    assert s["fits"] == 2
    gains = [f["true_gain"] for f in planned["fits"].values()]
    assert s["true_gain"]["min"] == min(gains)
    assert s["true_gain_below_zero"] == sum(g < 0 for g in gains)


def test_the_summary_takes_medians_over_runs() -> None:
    def run(projected, true, best):
        return {
            "fits": {
                "v": {
                    "projected_gain_share": projected,
                    "true_gain_share": true,
                    "true_gain": true * 100,
                }
            },
            "true_optimum": {
                "true_gain": best * 100,
                "true_gain_share": best,
                "steady_gain_share": best / 2,
            },
            "recommendation": {
                "threshold": 0.01,
                "change": true > 0.005,
                "true_gain": true * 100 if true > 0.005 else 0.0,
                "true_gain_share": true if true > 0.005 else 0.0,
            },
        }

    runs = [
        run(0.10, 0.02, 0.04),
        run(0.50, -0.01, 0.05),
        run(0.20, 0.003, 0.03),  # a gain of $0.30 a day is small but not a loss
        run(0.15, 0.01, 0.0),  # nothing to gain: left out of the share captured
    ]
    s = optimise.summarise(runs)
    assert s["v"] == {
        "projected_share": pytest.approx(0.175),  # the median, not the mean (0.2375)
        "true_share": pytest.approx(0.0065),
        "true_below_zero": 1,
        "captured": pytest.approx(0.1),  # of 0.5, -0.2 and 0.1
        "runs": 4,
    }
    assert s["true_optimum"] == {
        "true_share": pytest.approx(0.035),
        "steady_share": pytest.approx(0.0175),
        "runs": 4,
    }
    # Kept on runs 2 and 3 (gain 0), changed on 1 and 4; the 4th has nothing to gain.
    assert s["recommended"] == {
        "threshold": 0.01,
        "changes": 2,
        "true_share": pytest.approx(0.005),
        "true_below_zero": 0,
        "captured": pytest.approx(0.0),  # of 0.5, 0 and 0
        "runs": 4,
    }


def test_a_change_within_the_grid_s_step_is_no_move() -> None:
    def fit(x):
        return {"spend": {"a": x}, "projected_gain": 0.0, "true_gain": 0.0}

    result = {
        "current": {"a": 100.0},
        "resolution": 1.0,
        "true_optimum": {"spend": {"a": 100.0}},
        "fits": {"up": fit(101.5), "same": fit(100.6), "down": fit(98.0), "close": fit(99.2)},
    }
    s = optimise.spread(result)["channels"]["a"]
    assert (s["up"], s["down"]) == (1, 1)


@pytest.mark.parametrize(
    "kwargs, match",
    [({"floor": 1.2}, "floor"), ({"cap": 0.8}, "cap"), ({"floor": -0.1}, "floor"),
     ({"fits": ("true_curves",)}, "unknown fits"), ({"fits": ()}, "unknown fits")],
)  # fmt: skip
def test_a_bad_plan_is_refused(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        optimise.plan(5, days=400, **kwargs)


def test_the_command_line_writes_every_figure(tmp_path, capsys) -> None:
    out = tmp_path / "plan.json"
    args = ["--seeds", "5", "--days", "400", "--steps", "500", "--fits", "committed"]
    assert optimise.main([*args, "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "true optimum" in printed and "projected gain" in printed
    assert "held-out 91 days from 2024-11-05" in printed
    assert "steady optimum" in printed and "true optimum " not in printed.split("held-out")[1]
    figures = json.loads(out.read_text(encoding="utf-8"))
    assert [r["seed"] for r in figures["runs"]] == [5]
    assert set(figures["summary"]) == {
        "committed",
        "recommended",
        "by_threshold",
        "holdout",
        "true_optimum",
    }


def test_runs_fitted_at_once_give_the_same_figures(tmp_path, capsys) -> None:
    common = ["--seeds", "5,6", "--days", "400", "--steps", "500", "--fits", "committed"]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert optimise.main([*common, "--json", str(a)]) == 0
    assert optimise.main([*common, "--workers", "2", "--json", str(b)]) == 0
    assert "over 2 runs" in capsys.readouterr().out
    ra, rb = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    for r in (*ra["runs"], *rb["runs"]):
        for f in r["fits"].values():
            f.pop("seconds")
    assert ra == rb


@pytest.mark.parametrize(
    "args, message",
    [(["--workers", "0"], "workers"), (["--fits", "magic"], "unknown fits"),
     (["--floor", "2"], "floor")],
)  # fmt: skip
def test_the_command_line_refuses_bad_options(args, message, capsys) -> None:
    with pytest.raises(SystemExit):
        optimise.main(["--days", "400", *args])
    assert message in capsys.readouterr().err


def test_an_infinite_slope_at_no_spend_is_written_as_null(tmp_path, monkeypatch) -> None:
    # A Hill shape below 1 rises infinitely steeply from 0; a fit pinned there by its
    # bounds must still give valid JSON.
    curves = {"a": s_curve(100, 50, 0.6), "b": log_curve(500, 100)}
    assert optimise.allocate(curves, 100, upper={"a": 0.0}).marginal["a"] == float("inf")

    real = optimise.allocate

    def pinned(curves, total, lower=None, upper=None, **kwargs):
        upper = dict(upper or {}) | {"display": 0.0}
        lower = dict(lower or {}) | {"display": 0.0}
        return real(curves, total, lower, upper, **kwargs)

    monkeypatch.setattr(optimise, "allocate", pinned)
    monkeypatch.setattr(optimise.mmm, "fit", _fit_with_display_shape(0.6))
    out = tmp_path / "plan.json"
    args = ["--seeds", "5", "--days", "400", "--steps", "500", "--fits", "committed"]
    assert optimise.main([*args, "--json", str(out)]) == 0
    fit = json.loads(out.read_text(encoding="utf-8"))["runs"][0]["fits"]["committed"]
    assert fit["marginal"]["display"] is None and fit["spend"]["display"] == 0.0


def _fit_with_display_shape(shape: float):
    real = mmm.fit

    def fit(inputs, **kwargs):
        model = real(inputs, **kwargs, max_evals=50)
        model.channels = [
            dataclasses.replace(ch, shape=shape) if ch.name == "display" else ch
            for ch in model.channels
        ]
        return model

    return fit


@pytest.mark.parametrize(
    "kwargs, match",
    [({"holdout_days": 400}, "holdout_days"), ({"holdout_days": -1}, "holdout_days"),
     ({"floor": 1.0, "cap": 1.0}, "nothing to reallocate"), ({"cap": float("inf")}, "finite")],
)  # fmt: skip
def test_a_plan_it_cannot_make_is_refused_plainly(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        optimise.plan(5, days=400, fits=("committed",), **kwargs)


def test_the_gains_are_over_steady_budgets_and_the_steady_gap_is_reported(planned) -> None:
    # Spending the current budgets every day at their mean differs from how they were
    # spent (flights, quarterly levels); the true revenue of both is reported.
    data = generate(GeneratorConfig(seed=5, days=400))
    cd = data.channel_daily
    spend = np.zeros((4, 400))
    spend[cd["channel"], cd["day"]] = cd["spend_cents"] / 100
    actual = 0.0
    for c, ch in enumerate(data.config.channels):
        carry, a = spend[c, 0], np.empty(400)  # the generator's start: as if steady
        for t in range(400):
            carry = (1 - ch.decay) * spend[c, t] + ch.decay * carry
            a[t] = carry
        k = ch.half_saturation * ch.daily_spend
        actual += (data.betas[ch.name] * 90.0 * hill(a, k, ch.hill_shape))[:309].mean()
    t = planned["true_optimum"]
    assert t["true_actual"] == pytest.approx(actual, rel=1e-9)
    assert t["steady_gain"] == pytest.approx(t["true_now"] - actual)
    assert t["steady_gain_share"] == pytest.approx(
        t["steady_gain"] / planned["new_customer_revenue"]
    )
    summary = optimise.summarise([planned])["true_optimum"]
    assert summary["steady_share"] == pytest.approx(t["steady_gain_share"])


@pytest.mark.parametrize("truth, shape, half", [("moved", 2.5, 2.5), ("low", 0.7, 0.5)])
def test_a_truth_away_from_the_priors_is_generated_and_scored(truth, shape, half) -> None:
    got = optimise.plan(5, fits=("committed",), days=400, steps=500, truth=truth)
    assert got["truth"] == truth
    data = generate(identify.moved(GeneratorConfig(seed=5, days=400), shape, half))
    assert all((c.hill_shape, c.half_saturation) == (shape, half) for c in data.config.channels)
    current = got["current"]

    def true_revenue(alloc) -> float:
        return sum(
            data.betas[ch.name] * 90.0 * hill(alloc[ch.name], half * ch.daily_spend, shape)
            for ch in data.config.channels
        )

    f = got["fits"]["committed"]
    assert f["true_gain"] == pytest.approx(true_revenue(f["spend"]) - true_revenue(current))
    default = optimise.plan(5, fits=("committed",), days=400, steps=500)
    assert default["truth"] == "default"
    assert default["true_optimum"]["true_gain"] != pytest.approx(got["true_optimum"]["true_gain"])


def test_an_unknown_truth_is_refused(capsys) -> None:
    with pytest.raises(ValueError, match="unknown truth"):
        optimise.plan(5, days=400, truth="flat")
    with pytest.raises(SystemExit):
        optimise.main(["--days", "400", "--truth", "flat"])
    assert "invalid choice" in capsys.readouterr().err


def test_the_command_line_names_the_truth(tmp_path, capsys) -> None:
    out = tmp_path / "low.json"
    args = ["--seeds", "5", "--days", "400", "--steps", "500", "--fits", "committed"]
    assert optimise.main([*args, "--truth", "low", "--json", str(out)]) == 0
    assert "low truth: shape 0.7, half-saturation 0.5x" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["runs"][0]["truth"] == "low"


# --- a split robust to the fits, or no change -------------------------------------------------


def test_the_robust_split_is_the_one_every_fit_expects_most_from() -> None:
    current = {"a": 100.0, "b": 100.0}
    fits = {
        "f1": {"a": log_curve(1000, 50), "b": log_curve(100, 50)},  # a is worth more
        "f2": {"a": log_curve(170, 50), "b": log_curve(140, 50)},  # little in it
    }
    candidates = {"all_in_a": {"a": 190.0, "b": 10.0}, "a_bit": {"a": 110.0, "b": 90.0}}
    name, split, worst = optimise.robust_choice(fits, candidates, current)

    def gain(f, x):
        return optimise.steady_revenue(fits[f], x) - optimise.steady_revenue(fits[f], current)

    # all_in_a gains most under f1 but loses under f2; a_bit gains under both.
    assert gain("f1", candidates["all_in_a"]) > gain("f1", candidates["a_bit"])
    assert gain("f2", candidates["all_in_a"]) < 0 < gain("f2", candidates["a_bit"])
    assert (name, split) == ("a_bit", candidates["a_bit"])
    assert worst == pytest.approx(min(gain("f1", split), gain("f2", split)))


def test_when_every_candidate_loses_under_some_fit_the_budget_stays() -> None:
    current = {"a": 100.0, "b": 100.0}
    fits = {
        "f1": {"a": log_curve(1000, 50), "b": log_curve(100, 50)},
        "f2": {"a": log_curve(100, 50), "b": log_curve(1000, 50)},
    }
    candidates = {"more_a": {"a": 150.0, "b": 50.0}, "more_b": {"a": 50.0, "b": 150.0}}
    assert optimise.robust_choice(fits, candidates, current) == ("current", current, 0.0)


def test_a_change_is_recommended_only_above_the_threshold() -> None:
    result = {
        "current": {"a": 100.0},
        "robust": {
            "from": "f1",
            "spend": {"a": 120.0},
            "worst_projected_gain": 30.0,
            "worst_projected_share": 0.01,
            "true_gain": 12.0,
            "true_gain_share": 0.004,
        },
    }
    assert optimise.recommend(result, 0.01)["change"] is True  # at the threshold
    kept = optimise.recommend(result, 0.0101)
    assert kept == {
        "threshold": 0.0101,
        "change": False,
        "from": "current",
        "spend": {"a": 100.0},
        "true_gain": 0.0,
        "true_gain_share": 0.0,
    }
    result["robust"]["from"] = "current"
    assert optimise.recommend(result, 0.0)["change"] is False


def test_the_plan_reports_the_robust_split_and_its_recommendation(planned) -> None:
    r, rec = planned["robust"], planned["recommendation"]
    assert r["from"] in {"current", *planned["fits"]}
    if r["from"] != "current":
        assert r["spend"] == planned["fits"][r["from"]]["spend"]
    assert r["worst_projected_gain"] >= 0
    assert r["worst_projected_share"] == pytest.approx(
        r["worst_projected_gain"] / planned["new_customer_revenue"]
    )
    assert rec["threshold"] == optimise.THRESHOLD
    assert rec["change"] == (
        r["from"] != "current" and r["worst_projected_share"] >= rec["threshold"]
    )
    high = optimise.plan(5, fits=("committed", "lift"), days=400, steps=500, threshold=0.99)
    assert high["recommendation"]["change"] is False and high["recommendation"]["true_gain"] == 0
    assert high["robust"] == planned["robust"]


@pytest.mark.parametrize("bad", [-0.01, float("inf"), float("nan"), True])
def test_a_bad_threshold_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="threshold"):
        optimise.plan(5, days=400, fits=("committed",), threshold=bad)


def test_the_command_line_prints_the_recommendation(capsys) -> None:
    args = ["--seeds", "5,6", "--days", "400", "--steps", "500", "--fits", "committed,lift"]
    assert optimise.main([*args, "--threshold", "0.99"]) == 0
    out = capsys.readouterr().out
    assert out.count("keep the current budget") == 2
    assert "changed the budget on 0 of 2 runs, threshold 99.0%" in out


def test_the_summary_reports_each_candidate_threshold(planned) -> None:
    s = optimise.summarise([planned])["by_threshold"]
    assert list(s) == ["0", "0.005", "0.01"]
    for key, t in zip(s, optimise.REPORTED_THRESHOLDS, strict=True):
        rec = optimise.recommend(planned, t)
        assert s[key]["threshold"] == t and s[key]["changes"] == int(rec["change"])
        assert s[key]["true_share"] == pytest.approx(rec["true_gain_share"])
    worst = planned["robust"]["worst_projected_share"]
    assert s["0.01"]["changes"] == int(planned["robust"]["from"] != "current" and worst >= 0.01)


def test_each_reported_threshold_decides_on_its_own() -> None:
    run = {
        "current": {"a": 100.0},
        "true_optimum": {"true_gain": 20.0, "true_gain_share": 0.02, "steady_gain_share": 0.0},
        "fits": {"f": {"projected_gain_share": 0.01, "true_gain_share": 0.01, "true_gain": 10.0}},
        "robust": {
            "from": "f",
            "spend": {"a": 120.0},
            "worst_projected_gain": 7.0,
            "worst_projected_share": 0.007,
            "true_gain": 10.0,
            "true_gain_share": 0.01,
        },
    }
    run["recommendation"] = optimise.recommend(run, 0.0)
    s = optimise.summarise([run])["by_threshold"]
    assert [s[k]["changes"] for k in ("0", "0.005", "0.01")] == [1, 1, 0]
    assert [s[k]["captured"] for k in ("0", "0.005", "0.01")] == [0.5, 0.5, 0.0]


def test_the_default_threshold_is_the_one_chosen_on_the_dev_runs() -> None:
    assert optimise.THRESHOLD == 0.01
    assert optimise.plan.__kwdefaults__["threshold"] == 0.01


# --- each split on the held-out days ----------------------------------------------------------


def test_every_split_is_tried_on_the_held_out_days(planned) -> None:
    h = planned["holdout"]
    assert (h["days"], h["first_day"]) == (91, "2024-11-05")
    assert list(h["splits"]) == [*planned["fits"], "recommendation", "true_optimum"]
    assert list(h["forecast"]) == list(planned["fits"])
    data = generate(GeneratorConfig(seed=5, days=400))
    inputs = mmm.inputs_from_dataset(data)
    assert h["spend"] == pytest.approx(inputs.spend[:, 309:].sum())
    assert h["new_customer_revenue"] == pytest.approx(inputs.new_revenue[309:].sum())
    curves = {
        ch.name: ResponseCurve(
            "hill", data.betas[ch.name] * 90.0, ch.half_saturation * ch.daily_spend, ch.hill_shape
        )
        for ch in data.config.channels
    }
    decays = {ch.name: ch.decay for ch in data.config.channels}
    before = holdout.true_media_revenue(curves, decays, inputs.spend)[309:].sum()
    splits = {name: f["spend"] for name, f in planned["fits"].items()}
    splits["recommendation"] = planned["recommendation"]["spend"]
    splits["true_optimum"] = planned["true_optimum"]["spend"]
    for label, split in splits.items():
        new = holdout.reallocated(inputs.spend, 309, inputs.channels, planned["current"], split)
        true = holdout.true_media_revenue(curves, decays, new)[309:].sum() - before
        assert h["splits"][label]["true"] == pytest.approx(true, rel=1e-9, abs=1e-6)
        assert sum(h["splits"][label]["spend"].values()) == pytest.approx(h["spend"])
    if not planned["recommendation"]["change"]:
        assert h["splits"]["recommendation"]["true"] == pytest.approx(0, abs=1e-6)


def test_the_summary_reports_the_held_out_lifts(planned) -> None:
    h = planned["holdout"]
    s = optimise.summarise([planned])["holdout"]
    assert s["runs"] == 1 and list(s["fits"]) == list(planned["fits"])
    for name, v in s["fits"].items():
        own = h["splits"][name]
        assert v["projected_share"] == pytest.approx(own["projected_share"][name])
        assert v["true_share"] == pytest.approx(own["true_share"])
        assert v["true_below_zero"] == int(own["true"] < 0)
        assert v["forecast_error"] == pytest.approx(h["forecast"][name]["error_share"])
        assert v["forecast_error_abs"] == pytest.approx(abs(h["forecast"][name]["error_share"]))
    rec = h["splits"]["recommendation"]
    assert s["recommendation"]["worst_projected_share"] == pytest.approx(
        min(rec["projected_share"].values())
    )
    assert s["recommendation"]["true_roi"] == pytest.approx(rec["true_roi"])
    assert s["true_optimum"]["true_share"] == pytest.approx(
        h["splits"]["true_optimum"]["true_share"]
    )


def test_without_held_out_days_there_is_no_holdout_check() -> None:
    got = optimise.plan(5, fits=("committed",), days=400, steps=500, holdout_days=0)
    assert "holdout" not in got
    assert "holdout" not in optimise.summarise([got])


def test_a_run_without_held_out_revenue_is_left_out_of_the_shares(planned, capsys) -> None:
    empty = copy.deepcopy(planned)
    h = empty["holdout"]
    for f in h["forecast"].values():
        f["error_share"] = None
    for s in h["splits"].values():
        s["true_share"] = None
        s["projected_share"] = dict.fromkeys(s["projected_share"])
    s = optimise.summarise([planned, empty])["holdout"]
    one = optimise.summarise([planned])["holdout"]
    assert s["runs"] == 2
    for name, v in s["fits"].items():
        for key in ("projected_share", "true_share", "forecast_error", "forecast_error_abs"):
            assert v[key] == pytest.approx(one["fits"][name][key])
    assert s["recommendation"]["true_share"] == pytest.approx(one["recommendation"]["true_share"])
    nothing = optimise.summarise([empty])["holdout"]
    assert nothing["true_optimum"]["true_share"] is None
    optimise._print_holdout(h)
    optimise._print_holdout_summary(nothing)
    assert "n/a" in capsys.readouterr().out
