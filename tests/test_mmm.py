import dataclasses
import datetime as dt
import itertools
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from marketing_roi import mmm
from marketing_roi.adstock import geometric_adstock
from marketing_roi.generate import GeneratorConfig, generate, write
from marketing_roi.saturation import hill

START = np.datetime64("2024-01-01")


@pytest.fixture(scope="module")
def small():
    return generate(GeneratorConfig(seed=11, start=dt.date(2025, 10, 1), days=120))


@pytest.fixture(scope="module")
def generated(small, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("generated")
    write(small, out)
    return out


@pytest.fixture
def files(generated, tmp_path) -> Path:
    """A copy of the generated files that a test may damage."""
    return Path(shutil.copytree(generated, tmp_path / "data"))


def synthetic(days: int = 500, noise: float = 0.0, seed: int = 0):
    """Two channels and a known baseline; revenue = baseline + each channel's curve."""
    rng = np.random.default_rng(seed)
    dates = START + np.arange(days)
    on = (np.arange(days) // 30) % 3 != 2  # flighted: off one month in three
    spend = np.vstack(
        [
            900 * rng.lognormal(0, 0.35, days),
            400 * rng.lognormal(0, 0.5, days) * on,
        ]
    )
    true = {"a": (0.3, 800.0, 1.4), "b": (0.6, 300.0, 1.0)}
    betas = {"a": 2500.0, "b": 1200.0}
    x, names = mmm.baseline_design(dates)
    coef = np.zeros(len(names))
    coef[names.index("intercept")] = 3000
    coef[names.index("trend")] = 400
    coef[names.index("cos1")] = 500
    coef[names.index("sat")] = 150
    coef[names.index("holiday")] = 1000
    y = x @ coef
    for c, name in enumerate(("a", "b")):
        d, k, s = true[name]
        y = y + betas[name] * hill(geometric_adstock(spend[c], d), k, s)
    y = y * rng.lognormal(0, noise, days) if noise else y
    inputs = mmm.Inputs(dates, ("a", "b"), spend, y, y)
    return inputs, true, betas, coef


# --- inputs ----------------------------------------------------------------------------------


def test_inputs_from_files_equal_inputs_from_the_dataset(small, generated) -> None:
    a, b = mmm.inputs_from_dataset(small), mmm.read_inputs(generated)
    assert a.channels == b.channels
    np.testing.assert_array_equal(a.dates, b.dates)
    np.testing.assert_allclose(a.spend, b.spend, rtol=0, atol=1e-9)
    np.testing.assert_allclose(a.revenue, b.revenue, rtol=0, atol=1e-6)
    np.testing.assert_allclose(a.new_revenue, b.new_revenue, rtol=0, atol=1e-6)


def test_revenue_is_every_purchase_and_new_revenue_each_customer_s_first(small) -> None:
    inputs = mmm.inputs_from_dataset(small)
    s = small.sessions
    assert inputs.revenue.sum() == pytest.approx(s["revenue_cents"].sum() / 100, rel=1e-12)
    bought = s["purchase"] >= 0
    first_value = {}
    for user, cents in zip(s["user"][bought], s["revenue_cents"][bought], strict=True):
        first_value.setdefault(user, cents)  # sessions are in time order
    assert inputs.new_revenue.sum() == pytest.approx(sum(first_value.values()) / 100, rel=1e-12)
    # Repeat purchases exist, so new revenue is less than all revenue, never more.
    assert inputs.new_revenue.sum() < inputs.revenue.sum()
    assert (inputs.new_revenue <= inputs.revenue + 1e-9).all()


def test_a_user_s_first_purchase_is_found_in_a_file_out_of_time_order(generated, files) -> None:
    path = files / "ga4_sessions.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(lines[0] + "".join(reversed(lines[1:])), encoding="utf-8")
    a, b = mmm.read_inputs(generated), mmm.read_inputs(files)
    np.testing.assert_allclose(a.new_revenue, b.new_revenue, rtol=0, atol=1e-6)


def test_a_missing_day_in_the_spend_is_refused(files) -> None:
    path = files / "channel_daily.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:5] + lines[6:]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing a day"):
        mmm.read_inputs(files)


def test_a_purchase_outside_the_spend_s_days_is_refused(files) -> None:
    path = files / "ga4_sessions.csv"
    text = path.read_text(encoding="utf-8")
    first_purchase = next(
        line for line in text.splitlines()[1:] if line.endswith(tuple("0123456789"))
    )
    moved = first_purchase.replace(first_purchase.split(",")[1][:4], "2031", 1)
    path.write_text(text.replace(first_purchase, moved, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        mmm.read_inputs(files)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda i: {"dates": i.dates[::-1]}, "consecutive"),
        (lambda i: {"dates": i.dates[:1], "spend": i.spend[:, :1], "revenue": i.revenue[:1],
                    "new_revenue": i.new_revenue[:1]}, "consecutive"),
        (lambda i: {"spend": i.spend[:1]}, "spend must be"),
        (lambda i: {"revenue": i.revenue[:-1]}, "revenue"),
        (lambda i: {"spend": -i.spend}, "spend must be finite"),
        (lambda i: {"revenue": np.where(np.arange(len(i)) == 3, np.nan, i.revenue)}, "revenue"),
        (lambda i: {"new_revenue": i.revenue * 1.01}, "exceed"),
        (lambda i: {"channels": ("a", "a")}, "distinct"),
    ],
)  # fmt: skip
def test_inconsistent_inputs_are_refused(change, match) -> None:
    inputs, *_ = synthetic(60)
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(inputs, **change(inputs))


def test_target_picks_the_revenue_series() -> None:
    inputs, *_ = synthetic(60)
    inputs = dataclasses.replace(inputs, new_revenue=inputs.revenue / 2)
    assert inputs.target("total") is inputs.revenue
    assert inputs.target("new_customer") is inputs.new_revenue
    with pytest.raises(ValueError, match="target"):
        inputs.target("repeat")


# --- design ----------------------------------------------------------------------------------


def test_the_baseline_has_trend_seasons_weekdays_and_the_holiday_window() -> None:
    dates = np.datetime64("2024-01-01") + np.arange(731)
    x, names = mmm.baseline_design(dates, fourier_order=3)
    assert names == ["intercept", "trend", "sin1", "cos1", "sin2", "cos2", "sin3", "cos3",
                     "mon", "tue", "wed", "thu", "fri", "sat", "holiday"]  # fmt: skip
    col = dict(zip(names, x.T, strict=True))
    assert (col["intercept"] == 1).all()
    assert col["trend"][0] == 0 and col["trend"][366] == pytest.approx(366 / 365.25)
    # 2024-01-01 is a Monday; Sunday is the reference day.
    assert col["mon"][0] == 1 and x[6, names.index("mon") : names.index("holiday")].sum() == 0
    holiday = dates[col["holiday"] == 1].astype(str).tolist()
    assert holiday == ["2024-11-29", "2024-11-30", "2024-12-01", "2024-12-02",
                       "2025-11-28", "2025-11-29", "2025-11-30", "2025-12-01"]  # fmt: skip
    assert np.linalg.matrix_rank(x) == x.shape[1]


# --- solver and search -----------------------------------------------------------------------


def test_media_coefficients_are_the_best_non_negative_least_squares() -> None:
    rng = np.random.default_rng(4)
    base = np.column_stack([np.ones(200), rng.normal(size=200)])
    media = rng.random((200, 3))
    # The third column hurts sales here, so the unconstrained fit gives it a negative
    # coefficient.
    y = base @ [5.0, 1.0] + media @ [2.0, 0.5, -3.0] + rng.normal(0, 0.1, 200)
    coef, sse = mmm._solve(base, media, y)
    assert (coef[2:] >= 0).all() and coef[4] == 0
    # Against every subset by brute force.
    best = np.inf
    for keep in itertools.chain.from_iterable(
        itertools.combinations(range(3), r) for r in range(4)
    ):
        x = np.hstack([base, media[:, list(keep)]])
        c, *_ = np.linalg.lstsq(x, y, rcond=None)
        if (c[2:] >= 0).all():
            best = min(best, ((y - x @ c) ** 2).sum())
    assert sse == pytest.approx(best, rel=1e-12)
    assert sse == pytest.approx(((y - np.hstack([base, media]) @ coef) ** 2).sum(), rel=1e-12)


def test_unconstrained_coefficients_are_kept_when_none_is_negative() -> None:
    rng = np.random.default_rng(5)
    base, media = np.ones((50, 1)), rng.random((50, 2))
    y = 1 + media @ [2.0, 3.0] + rng.normal(0, 0.01, 50)
    coef, _ = mmm._solve(base, media, y)
    expected, *_ = np.linalg.lstsq(np.hstack([base, media]), y, rcond=None)
    np.testing.assert_allclose(coef, expected, rtol=1e-12)


def test_nelder_mead_finds_the_minimum_of_a_valley() -> None:
    def rosenbrock(p):
        return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

    x, value, evals = mmm.nelder_mead(rosenbrock, np.array([-1.2, 1.0]), tol=1e-14)
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-4)
    assert value < 1e-8 and evals < 4000


def test_nelder_mead_does_not_stop_either_side_of_the_minimum() -> None:
    # Its first points, 99.5 and 100.5 on the way in, have equal values; agreeing values
    # alone would stop there.
    x, value, _ = mmm.nelder_mead(lambda p: (p[0] - 100) ** 2, np.array([0.0]))
    assert x[0] == pytest.approx(100, abs=1e-5) and value < 1e-10


def test_nelder_mead_expands_its_step_towards_a_distant_minimum() -> None:
    # Expansion doubles the step, so a minimum 10,000 steps away takes few evaluations.
    _, _, evals = mmm.nelder_mead(lambda p: (p[0] - 5000) ** 2, np.array([0.0]))
    assert evals < 200


def test_the_grid_keeps_its_best_point() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=7)
    model = mmm.fit(inputs, holdout_days=50, max_evals=0)
    n = model.train_days
    base = mmm.baseline_design(inputs.dates)[0][:n]
    scale = inputs.spend[:, :n].mean(axis=1)
    start = np.column_stack([np.full(2, 0.4), scale, np.ones(2)])  # a point on the grid
    start_sse = mmm._solve(base, mmm._media(inputs.spend[:, :n], start), inputs.revenue[:n])[1]
    assert model.search["grid_sse"] < start_sse
    assert model.sse <= model.search["grid_sse"]  # the simplex starts from the grid's point


def test_the_search_coordinates_round_trip_within_the_bounds() -> None:
    scale = np.array([900.0, 300.0])
    params = np.array([[0.3, 800.0, 1.4], [0.6, 300.0, 1.0]])
    np.testing.assert_allclose(mmm._from_unit(mmm._to_unit(params, scale), scale), params)
    lo, hi = mmm._bounds(scale)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a simplex far out must not overflow
        far = mmm._from_unit(np.array([-1e4, 1e4, 50.0, 50.0, -50.0, -1e4]), scale)
    assert ((far >= lo * (1 - 1e-12)) & (far <= hi * (1 + 1e-12))).all()


# --- fit -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exact():
    inputs, true, betas, coef = synthetic()
    return inputs, true, betas, coef, mmm.fit(inputs, holdout_days=60)


def test_without_noise_the_fit_predicts_held_out_days(exact) -> None:
    inputs, *_, model = exact
    ev = mmm.evaluate(model, inputs)
    assert ev["train"]["r2"] > 0.9999 and ev["holdout"]["r2"] > 0.999
    assert ev["holdout"]["mape"] < 0.002
    assert model.train_days == 440 and ev["holdout"]["days"] == 60


def test_without_noise_the_fit_recovers_the_curves(exact) -> None:
    inputs, true, betas, coef, model = exact
    for ch in model.channels:
        d, k, s = true[ch.name]
        assert ch.decay == pytest.approx(d, abs=0.02)
        x = np.linspace(0, 2 * k, 50)
        np.testing.assert_allclose(ch.curve.response(x), betas[ch.name] * hill(x, k, s),
                                   rtol=0, atol=0.02 * betas[ch.name])  # fmt: skip
    parts = model.decompose(inputs)
    assert set(parts) == {"baseline", "a", "b"}
    np.testing.assert_allclose(sum(parts.values()), model.predict(inputs), rtol=1e-12)


def test_true_curves_given_as_fixed_are_used_as_they_are(exact) -> None:
    inputs, true, betas, coef, _ = exact
    model = mmm.fit(inputs, holdout_days=60, fixed=true)
    for ch in model.channels:
        assert (ch.decay, ch.half_saturation, ch.shape) == true[ch.name]
        assert ch.beta == pytest.approx(betas[ch.name], rel=1e-8)
    np.testing.assert_allclose(model.baseline_coef, coef, rtol=1e-8, atol=1e-6)
    assert model.sse < 1e-12 * (inputs.revenue**2).sum()


def test_the_held_out_days_never_reach_the_fit() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=2)
    changed = inputs.revenue.copy()
    changed[-50:] *= 3
    other = dataclasses.replace(inputs, revenue=changed, new_revenue=changed)
    a = mmm.fit(inputs, holdout_days=50, max_evals=300)
    b = mmm.fit(other, holdout_days=50, max_evals=300)
    assert a.channels == b.channels and a.sse == b.sse
    np.testing.assert_array_equal(a.baseline_coef, b.baseline_coef)
    assert mmm.evaluate(a, inputs)["holdout"] != mmm.evaluate(b, other)["holdout"]


def test_the_target_chooses_which_revenue_is_fitted() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=3)
    inputs = dataclasses.replace(inputs, new_revenue=inputs.revenue / 2)
    total = mmm.fit(inputs, holdout_days=50, target="total", max_evals=300)
    new = mmm.fit(inputs, holdout_days=50, max_evals=300)
    assert new.target == "new_customer" and total.target == "total"
    # Halving the revenue halves every coefficient and leaves the curves' shapes alone.
    for a, b in zip(total.channels, new.channels, strict=True):
        assert (a.decay, a.half_saturation, a.shape) == pytest.approx(
            (b.decay, b.half_saturation, b.shape), rel=1e-6
        )
        assert b.beta == pytest.approx(a.beta / 2, rel=1e-6)
    np.testing.assert_allclose(new.predict(inputs), total.predict(inputs) / 2, rtol=1e-6)
    baseline = mmm.fit_baseline_only(inputs, holdout_days=50)
    np.testing.assert_allclose(
        baseline, mmm.fit_baseline_only(inputs, holdout_days=50, target="total") / 2, rtol=1e-9
    )


def test_media_terms_explain_more_than_the_baseline_alone() -> None:
    inputs, *_ = synthetic(400, noise=0.08, seed=6)
    model = mmm.fit(inputs, holdout_days=60, max_evals=600)
    with_media = mmm.evaluate(model, inputs)
    alone = mmm.evaluate(model, inputs, mmm.fit_baseline_only(inputs, holdout_days=60))
    assert with_media["holdout"]["r2"] > alone["holdout"]["r2"]
    assert with_media["train"]["r2"] > alone["train"]["r2"]


@pytest.mark.parametrize("holdout", [-1, 1.5, True, "7", 450])
def test_a_bad_holdout_is_refused(holdout) -> None:
    inputs, *_ = synthetic(460)
    with pytest.raises(ValueError, match="holdout_days|too few"):
        mmm.fit(inputs, holdout_days=holdout)


def test_a_numpy_whole_number_is_a_holdout() -> None:
    inputs, *_ = synthetic(200)
    model = mmm.fit(inputs, holdout_days=np.int64(30), max_evals=50)
    assert model.holdout_days == 30 and type(model.holdout_days) is int


def test_fixed_curves_must_name_known_channels() -> None:
    inputs, *_ = synthetic(200)
    with pytest.raises(ValueError, match="'c'"):
        mmm.fit(inputs, fixed={"c": (0.1, 100.0, 1.0)})


def test_a_channel_that_never_spends_on_the_training_days_is_refused() -> None:
    inputs, *_ = synthetic(200)
    spend = inputs.spend.copy()
    spend[1, :150] = 0
    with pytest.raises(ValueError, match="some spend"):
        mmm.fit(dataclasses.replace(inputs, spend=spend), holdout_days=50)


def test_no_holdout_scores_the_training_days_only() -> None:
    inputs, *_ = synthetic(200)
    model = mmm.fit(inputs, holdout_days=0, max_evals=50)
    assert set(mmm.evaluate(model, inputs)) == {"train"}


# --- scores ----------------------------------------------------------------------------------


def test_scores_have_their_textbook_values() -> None:
    actual = np.array([100.0, 200.0, 300.0, 400.0])
    assert mmm.scores(actual, actual) == {"r2": 1.0, "mape": 0.0, "rmse": 0.0, "days": 4}
    s = mmm.scores(actual, np.full(4, 250.0))
    assert s["r2"] == pytest.approx(0.0) and s["rmse"] == pytest.approx(np.sqrt(12500))
    assert s["mape"] == pytest.approx((1.5 + 0.25 + 1 / 6 + 0.375) / 4)
    # Worse than the mean gives a negative R².
    assert mmm.scores(actual, actual[::-1])["r2"] == pytest.approx(-3.0)


def test_scores_refuse_what_they_cannot_score() -> None:
    with pytest.raises(ValueError, match="no days"):
        mmm.scores(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="above 0"):
        mmm.scores(np.array([1.0, 0.0]), np.array([1.0, 1.0]))


# --- comparison with the truth and the command line ------------------------------------------


def test_the_truth_comparison_reads_the_generators_record(small, generated) -> None:
    truth, caused = mmm.read_truth(generated)
    td = small.truth_daily
    for i, ch in enumerate(small.config.channels):
        assert caused[ch.name].sum() == pytest.approx(
            td["revenue_cents"][td["component"] == i].sum() / 100, rel=1e-12
        )
        assert mmm.true_params(truth)[ch.name] == (
            ch.decay, ch.half_saturation * ch.daily_spend, ch.hill_shape
        )  # fmt: skip


def test_the_truth_comparison_splits_train_and_held_out_days(small, generated) -> None:
    truth, caused = mmm.read_truth(generated)
    inputs = mmm.inputs_from_dataset(small)
    model = mmm.fit(inputs, holdout_days=30, max_evals=50)
    parts = model.decompose(inputs)
    for r in mmm.compare_with_truth(model, inputs, truth, caused):
        name = r["channel"]
        assert r["revenue_train"] == pytest.approx(
            (parts[name][:90].sum(), caused[name][:90].sum()), rel=1e-12
        )
        assert r["revenue_holdout"] == pytest.approx(
            (parts[name][90:].sum(), caused[name][90:].sum()), rel=1e-12
        )
        assert caused[name][90:].sum() > 0


def test_the_command_line_fits_and_writes_its_figures(generated, tmp_path, capsys) -> None:
    out = tmp_path / "mmm.json"
    args = ["--data", str(generated), "--holdout-days", "20", "--json", str(out)]
    assert mmm.main(args) == 0
    printed = capsys.readouterr().out
    assert "100 training days, 20 held out" in printed and "baseline terms only" in printed
    figures = json.loads(out.read_text(encoding="utf-8"))
    assert figures["target"] == "new_customer"
    assert set(figures["fit"]) == {"train", "holdout"}
    assert {r["channel"] for r in figures["channels"]} == {
        "paid_search", "paid_social", "display", "online_video"
    }  # fmt: skip
    assert figures["search"]["sse"] <= figures["search"]["grid_sse"]


def test_the_command_line_reports_a_damaged_folder(files, capsys) -> None:
    (files / "truth_daily.csv").unlink()
    assert mmm.main(["--data", str(files)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_the_command_line_refuses_a_holdout_longer_than_the_data(generated, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        mmm.main(["--data", str(generated), "--holdout-days", "500"])
    assert exc.value.code == 2
    assert "too few" in capsys.readouterr().err
