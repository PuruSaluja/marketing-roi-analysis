import argparse
import json

import numpy as np
import pytest

from marketing_roi import identify, mmm
from marketing_roi.generate import GeneratorConfig, generate


@pytest.fixture(scope="module")
def run():
    data = generate(GeneratorConfig(seed=5, days=400))
    return data, identify.caused_revenue(data)


def test_caused_revenue_is_each_channel_s_rows_of_the_truth(run) -> None:
    data, caused = run
    td = data.truth_daily
    for c, ch in enumerate(data.config.channels):
        rows = td["component"] == c
        assert caused[ch.name].sum() == pytest.approx(td["revenue_cents"][rows].sum() / 100)
        day = 123
        on_day = rows & (td["day"] == day)
        assert caused[ch.name][day] == pytest.approx(td["revenue_cents"][on_day].sum() / 100)
        assert len(caused[ch.name]) == 400


def test_the_lift_is_paid_search_s_caused_revenue_with_seeded_noise(run) -> None:
    data, caused = run
    lift = identify.lift_for(data, caused)
    true = caused["paid_search"][182:210].sum()
    assert lift.channel == "paid_search"
    assert (lift.start, lift.end) == (data.dates[182], data.dates[209])
    assert lift.se == pytest.approx(0.15 * true)
    e = np.random.default_rng([5, 2]).normal(0, 0.15)
    assert lift.revenue == pytest.approx(true * (1 + e))
    assert identify.lift_for(data, caused) == lift


def test_the_moved_truth_changes_only_the_curves() -> None:
    config = GeneratorConfig(seed=3)
    moved = identify.moved(config)
    for a, b in zip(config.channels, moved.channels, strict=True):
        assert (b.hill_shape, b.half_saturation) == (2.5, 2.5)
        assert (a.name, a.decay, a.daily_spend, a.cvr) == (b.name, b.decay, b.daily_spend, b.cvr)
    assert moved.seed == 3 and moved.days == config.days


def test_variants_become_fit_options(run) -> None:
    data, caused = run
    assert identify.options("committed", data, caused, 91) == {"holdout_days": 91}
    both = identify.options("prior_0.25+lift", data, caused, 91)
    assert both["prior"] == mmm.Prior(sd=0.25) and len(both["lifts"]) == 1
    assert identify.options("weekly", data, caused, 0) == {"holdout_days": 0, "weekly": True}
    fixed = identify.options("true_curves", data, caused, 91)["fixed"]
    assert fixed["online_video"] == (0.70, 800.0, 1.0)
    with pytest.raises(ValueError, match="unknown variant"):
        identify.options("prior_0.5+nothing", data, caused, 91)


def test_a_run_scores_each_curve_against_the_truth() -> None:
    out = identify.run_seed(5, ("true_curves", "committed"), False, 91, days=400)
    assert out["seed"] == 5 and set(out["variants"]) == {"true_curves", "committed"}
    data = generate(GeneratorConfig(seed=5, days=400))
    inputs = mmm.inputs_from_dataset(data)
    caused = identify.caused_revenue(data)
    model = mmm.fit(inputs, holdout_days=91, fixed=mmm.true_params(_truth(data)))
    parts = model.decompose(inputs)
    for c, ch in enumerate(data.config.channels):
        got = out["variants"]["true_curves"]["channels"][ch.name]
        x = inputs.spend[c, :309].mean()
        k, s = ch.half_saturation * ch.daily_spend, ch.hill_shape
        slope = s * x ** (s - 1) * k**s / (x**s + k**s) ** 2  # the Hill curve's derivative
        assert got["marginal_true"] == pytest.approx(data.betas[ch.name] * 90.0 * slope)
        fitted = model.channels[c]
        assert got["marginal_fitted"] == pytest.approx(fitted.beta * slope)
        assert got["marginal"] == pytest.approx(got["marginal_fitted"] / got["marginal_true"])
        assert got["credit"] == pytest.approx(
            parts[ch.name][:309].sum() / caused[ch.name][:309].sum()
        )
        assert (got["shape"], got["half_saturation"]) == pytest.approx((s, k / x))
    held = out["variants"]["committed"]
    assert 0 < held["holdout_mape"] < 1 and held["holdout_r2"] is not None


def _truth(data) -> dict:
    return {
        "channels": {
            ch.name: {
                "decay": ch.decay,
                "half_saturation_dollars": ch.half_saturation * ch.daily_spend,
                "hill_shape": ch.hill_shape,
            }
            for ch in data.config.channels
        }
    }


def _fake(credits: list[float], marginals: list[float], mape: float) -> dict:
    return {
        "variants": {
            "v": {
                "channels": {
                    "a": {"credit": credits[0], "marginal": marginals[0]},
                    "b": {"credit": credits[1], "marginal": marginals[1]},
                },
                "holdout_mape": mape,
            }
        }
    }


def test_the_summary_counts_seeds_within_a_quarter() -> None:
    runs = [
        _fake([1.25, 0.70], [0.75, 1.0], 0.10),  # a both at the edge; b credit out
        _fake([1.26, 1.00], [1.00, 1.0], 0.20),  # a credit just out; b both
        _fake([0.90, 1.00], [1.30, 1.2], 0.30),  # a marginal out; b both
    ]
    s = identify.summarise(runs, ("v",))["v"]
    assert s["channels"]["a"] == {
        "credit_within": 2,
        "marginal_within": 2,
        "both_within": 1,
        "credit_range": [0.90, 1.26],
        "marginal_range": [0.75, 1.30],
    }
    assert (s["channels"]["b"]["credit_within"], s["channels"]["b"]["both_within"]) == (2, 2)
    assert s["both_within_pairs"] == 3 and s["seeds"] == 3
    assert s["median_holdout_mape"] == pytest.approx(0.20)


@pytest.mark.parametrize(
    "text, seeds", [("10-19", list(range(10, 20))), ("3,5,8", [3, 5, 8]), ("1-2,7", [1, 2, 7]),
                    ("4", [4])]
)  # fmt: skip
def test_seed_lists(text, seeds) -> None:
    assert identify.seed_list(text) == seeds


@pytest.mark.parametrize("text", ["5-3", "1-3,2", "a", ""])
def test_bad_seed_lists_are_refused(text) -> None:
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        identify.seed_list(text)


def test_the_command_line_writes_every_figure(tmp_path, capsys) -> None:
    out = tmp_path / "results" / "figures.json"
    args = ["--seeds", "5", "--days", "400", "--variants", "committed,lift", "--json", str(out)]
    assert identify.main(args) == 0
    printed = capsys.readouterr().out
    assert "committed" in printed and "lift" in printed and "paid_search" in printed
    figures = json.loads(out.read_text(encoding="utf-8"))
    assert figures["seeds"] == [5] and figures["days"] == 400 and figures["within"] == 0.25
    assert set(figures["summary"]) == {"committed", "lift"}
    assert figures["runs"][0]["variants"]["lift"]["channels"]["paid_search"]["credit"] > 0


@pytest.mark.parametrize(
    "args, message",
    [(["--seeds", "1", "--variants", "committed,magic"], "unknown variants"),
     (["--seeds", "1", "--variants", ","], "unknown variants"),
     (["--seeds", "1", "--workers", "0"], "workers")],
)  # fmt: skip
def test_the_command_line_refuses_bad_options(args, message, capsys) -> None:
    with pytest.raises(SystemExit):
        identify.main(args)
    assert message in capsys.readouterr().err


def test_seeds_fitted_at_once_give_the_same_figures(tmp_path) -> None:
    serial, parallel = tmp_path / "serial.json", tmp_path / "parallel.json"
    common = ["--seeds", "5,6", "--days", "400", "--variants", "committed"]
    assert identify.main([*common, "--json", str(serial)]) == 0
    assert identify.main([*common, "--workers", "2", "--json", str(parallel)]) == 0
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (serial, parallel))
    for run in (*a["runs"], *b["runs"]):
        for v in run["variants"].values():
            v.pop("seconds")
    assert a["runs"] == b["runs"] and [r["seed"] for r in b["runs"]] == [5, 6]
