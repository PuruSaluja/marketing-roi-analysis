import dataclasses
import itertools
import json

import numpy as np
import pytest

from marketing_roi import mmm, optimise
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
            "true_optimum": {"true_gain": best * 100, "true_gain_share": best},
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
    assert s["true_optimum"] == {"true_share": pytest.approx(0.035), "runs": 4}


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
    figures = json.loads(out.read_text(encoding="utf-8"))
    assert [r["seed"] for r in figures["runs"]] == [5]
    assert set(figures["summary"]) == {"committed", "true_optimum"}


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
