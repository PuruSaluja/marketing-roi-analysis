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
    with warnings.catch_warnings():
        # numpy 2.5 deprecates adding a bare integer to a date, which will become an error.
        warnings.simplefilter("error")
        b = mmm.read_inputs(generated)
    a = mmm.inputs_from_dataset(small)
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


def replace_line(path: Path, line: int, text: str | None) -> None:
    """Replace one line of a file with ``text`` (which may hold several lines), or with
    None drop it."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if text is None:
        del lines[line]
    else:
        lines[line] = text
    path.write_text("".join(x + "\n" for x in lines), encoding="utf-8")


def line_of(path: Path, line: int = 1) -> str:
    return path.read_text(encoding="utf-8").splitlines()[line]


def duplicate_first_row(path: Path) -> None:
    replace_line(path, 1, line_of(path) + "\n" + line_of(path))


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        # A repeated row would silently replace the first.
        (lambda d: duplicate_first_row(d / "channel_daily.csv"), "two rows"),
        (lambda d: (d / "ga4_sessions.csv").write_text("", encoding="utf-8"), "empty"),
        (lambda d: replace_line(d / "channel_daily.csv", 0, "date,channel"), "no spend column"),
        (lambda d: replace_line(d / "ga4_sessions.csv", 3, "4,2025-10-01 00:01:00"), "line 4"),
        (lambda d: replace_line(d / "channel_daily.csv", 1, "2025-10-01,paid_search"), "line 2"),
        (lambda d: replace_line(d / "channel_daily.csv", 1, "2025-10-01,paid_search,x,1,1,0,0"),
         "could not convert"),
    ],
)  # fmt: skip
def test_a_damaged_input_file_is_refused(files, damage, match) -> None:
    damage(files)
    with pytest.raises(ValueError, match=match):
        mmm.read_inputs(files)


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        (lambda d: replace_line(d / "truth_daily.csv", -1, None), "119 of the inputs' 120 days"),
        (lambda d: duplicate_first_row(d / "truth_daily.csv"), "two rows"),
        (lambda d: replace_line(d / "truth_daily.csv", 1, "2031-01-01,paid_search,,,0,0.00"),
         "outside"),
    ],
)  # fmt: skip
def test_a_damaged_truth_file_is_refused(small, files, damage, match) -> None:
    damage(files)
    with pytest.raises(ValueError, match=match):
        mmm.read_truth(files, small.dates)


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


# --- priors, lifts and weekly sums -----------------------------------------------------------


def test_extra_rows_are_solved_with_the_rest() -> None:
    rng = np.random.default_rng(8)
    base, media = np.column_stack([np.ones(80), rng.normal(size=80)]), rng.random((80, 2))
    y = base @ [3.0, 1.0] + media @ [2.0, 1.0] + rng.normal(0, 0.1, 80)
    rows, targets = np.array([[40.0, 0.0]]), np.array([200.0])  # asks for a coefficient of 5
    coef, sse = mmm._solve(base, media, y, (rows, targets))
    x = np.vstack([np.hstack([base, media]), np.hstack([np.zeros((1, 2)), rows])])
    expected, *_ = np.linalg.lstsq(x, np.concatenate([y, targets]), rcond=None)
    np.testing.assert_allclose(coef, expected, rtol=1e-10)
    assert sse == pytest.approx(((np.concatenate([y, targets]) - x @ coef) ** 2).sum(), rel=1e-10)
    assert 2.0 < coef[2] < 5.0  # pulled from 2 towards the row's 5


def test_a_tight_prior_holds_the_curves_at_its_centre() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=7)
    prior = mmm.Prior(shape=1.3, half_saturation=0.7, sd=1e-4)
    model = mmm.fit(inputs, holdout_days=50, prior=prior, max_evals=400)
    scale = inputs.spend[:, : model.train_days].mean(axis=1)
    for c, ch in enumerate(model.channels):
        assert ch.shape == pytest.approx(1.3, rel=1e-3)
        assert ch.half_saturation == pytest.approx(0.7 * scale[c], rel=1e-3)


def test_the_prior_adds_its_penalty_to_the_training_error() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=7)
    model = mmm.fit(inputs, holdout_days=50, prior=mmm.Prior(sd=0.5), max_evals=400)
    n = model.train_days
    scale = inputs.spend[:, :n].mean(axis=1)
    penalty = sum(
        np.log(ch.shape) ** 2 + np.log(ch.half_saturation / scale[c]) ** 2
        for c, ch in enumerate(model.channels)
    )
    noise = model.search["noise_variance"]
    assert model.search["objective"] == pytest.approx(model.sse + noise * penalty / 0.25)
    # The noise is the baseline terms' residual variance, fixed before the search.
    base = mmm.baseline_design(inputs.dates)[0][:n]
    c0, *_ = np.linalg.lstsq(base, inputs.new_revenue[:n], rcond=None)
    resid = inputs.new_revenue[:n] - base @ c0
    # Degrees of freedom count the columns that vary: no Black Friday falls in these days.
    assert np.linalg.matrix_rank(base) == base.shape[1] - 1 and not base[:, -1].any()
    assert noise == pytest.approx(resid @ resid / (n - base.shape[1] + 1))
    # sse stays the training days' squared error.
    resid = (inputs.new_revenue - model.predict(inputs))[:n]
    assert model.sse == pytest.approx(resid @ resid, rel=1e-9)


def test_a_prior_does_not_bind_curves_given_as_fixed(exact) -> None:
    inputs, true, *_ = exact
    model = mmm.fit(inputs, holdout_days=60, fixed=true, prior=mmm.Prior(sd=1e-4))
    for ch in model.channels:
        assert (ch.decay, ch.half_saturation, ch.shape) == true[ch.name]
    assert model.search["objective"] == pytest.approx(model.sse)


@pytest.mark.parametrize(
    "field, value", [("shape", 0), ("sd", -1.0), ("half_saturation", float("inf")),
                     ("sd", float("nan")), ("shape", True), ("sd", "1")]
)  # fmt: skip
def test_a_bad_prior_is_refused(field, value) -> None:
    with pytest.raises(ValueError, match=f"prior {field}"):
        mmm.Prior(**{field: value})


def test_the_prior_must_be_a_prior() -> None:
    inputs, *_ = synthetic(200)
    with pytest.raises(TypeError, match="Prior"):
        mmm.fit(inputs, prior={"sd": 0.5})


def lift(inputs, channel: str, first: int, last: int, revenue: float, se: float) -> mmm.Lift:
    return mmm.Lift(channel, inputs.dates[first], inputs.dates[last], revenue, se)


def test_a_precise_lift_sets_the_channel_s_revenue_over_its_window() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=7)
    plain = mmm.fit(inputs, holdout_days=50, max_evals=400)
    wanted = 1.5 * plain.decompose(inputs)["a"][100:128].sum()
    model = mmm.fit(
        inputs,
        holdout_days=50,
        lifts=(lift(inputs, "a", 100, 127, wanted, 1e-6 * wanted),),
        max_evals=400,
    )
    assert model.decompose(inputs)["a"][100:128].sum() == pytest.approx(wanted, rel=1e-6)
    resid = (inputs.new_revenue - model.predict(inputs))[: model.train_days]
    assert model.sse == pytest.approx(resid @ resid, rel=1e-9)
    assert model.search["objective"] >= model.sse


def test_a_lift_adds_its_squared_error_in_standard_errors_times_the_noise() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=7)
    measured = lift(inputs, "b", 30, 79, 20_000.0, 2_000.0)
    model = mmm.fit(inputs, holdout_days=50, lifts=(measured,), max_evals=400)
    got = model.decompose(inputs)["b"][30:80].sum()
    z = (got - measured.revenue) / measured.se
    noise = model.search["noise_variance"]
    assert model.search["objective"] == pytest.approx(model.sse + noise * z**2, rel=1e-9)


def test_a_loose_lift_changes_little() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=7)
    plain = mmm.fit(inputs, holdout_days=50, max_evals=400)
    before = plain.decompose(inputs)["a"][100:128].sum()
    model = mmm.fit(
        inputs,
        holdout_days=50,
        lifts=(lift(inputs, "a", 100, 127, 1.5 * before, 1e6 * before),),
        max_evals=400,
    )
    np.testing.assert_allclose(model.predict(inputs), plain.predict(inputs), rtol=1e-3)


@pytest.mark.parametrize(
    "first, last, match",
    [(240, 260, "not within the training days"), (-1, 10, "not within"), (20, 10, "before")],
)
def test_a_lift_outside_the_training_days_is_refused(first, last, match) -> None:
    inputs, *_ = synthetic(300)
    dates = START + np.arange(-5, 300)  # day -1 is the day before the inputs start
    with pytest.raises(ValueError, match=match):
        mmm.fit(
            inputs,
            holdout_days=50,
            lifts=(mmm.Lift("a", dates[first + 5], dates[last + 5], 1000.0, 100.0),),
        )


@pytest.mark.parametrize(
    "revenue, se", [(-1.0, 1.0), (1.0, 0.0), (float("nan"), 1.0), (1.0, float("inf")),
                    (True, 1.0), (1.0, None)]
)  # fmt: skip
def test_a_bad_lift_is_refused(revenue, se) -> None:
    with pytest.raises(ValueError, match="lift (revenue|se)"):
        mmm.Lift("a", START, START + np.timedelta64(27, "D"), revenue, se)


def test_a_lift_must_name_a_known_channel() -> None:
    inputs, *_ = synthetic(200)
    with pytest.raises(ValueError, match="'c'"):
        mmm.fit(inputs, lifts=(lift(inputs, "c", 10, 20, 100.0, 10.0),))
    with pytest.raises(TypeError, match="Lift"):
        mmm.fit(inputs, lifts=(("a", 10, 20, 100.0, 10.0),))


def test_weekly_sums_recover_the_curves_without_noise() -> None:
    inputs, true, betas, _ = synthetic()
    model = mmm.fit(inputs, holdout_days=60, weekly=True)
    weekdays = [model.baseline_names.index(d) for d in mmm.WEEKDAYS]
    assert (model.baseline_coef[weekdays] == 0).all()
    weeks = mmm._weeks(len(inputs) - model.train_days)
    held = slice(model.train_days, None)
    actual, predicted = weeks @ inputs.new_revenue[held], weeks @ model.predict(inputs)[held]
    np.testing.assert_allclose(predicted, actual, rtol=0.005)
    for ch in model.channels:
        d, k, s = true[ch.name]
        x = np.linspace(0, 2 * k, 50)
        np.testing.assert_allclose(ch.curve.response(x), betas[ch.name] * hill(x, k, s),
                                   rtol=0, atol=0.03 * betas[ch.name])  # fmt: skip


def test_weekly_sums_leave_out_the_last_part_week() -> None:
    inputs, *_ = synthetic(300, noise=0.1, seed=2)
    # 250 training days are 35 weeks and 5 days; those 5 days must not reach the fit.
    changed = inputs.new_revenue.copy()
    changed[245:250] *= 3
    other = dataclasses.replace(inputs, revenue=changed, new_revenue=changed)
    a = mmm.fit(inputs, holdout_days=50, weekly=True, max_evals=300)
    b = mmm.fit(other, holdout_days=50, weekly=True, max_evals=300)
    assert a.channels == b.channels
    np.testing.assert_array_equal(a.baseline_coef, b.baseline_coef)
    assert mmm._weeks(16).tolist() == [[1.0] * 7 + [0.0] * 9, [0.0] * 7 + [1.0] * 7 + [0.0] * 2]


def test_weekly_sums_need_enough_weeks() -> None:
    inputs, *_ = synthetic(200)
    with pytest.raises(ValueError, match="training weeks"):
        mmm.fit(inputs, holdout_days=120, weekly=True)  # 80 days: 11 weeks


# --- scores ----------------------------------------------------------------------------------


def test_scores_have_their_textbook_values() -> None:
    actual = np.array([100.0, 200.0, 300.0, 400.0])
    assert mmm.scores(actual, actual) == {
        "r2": 1.0, "mape": 0.0, "mape_days": 4, "rmse": 0.0, "days": 4
    }  # fmt: skip
    s = mmm.scores(actual, np.full(4, 250.0))
    assert s["r2"] == pytest.approx(0.0) and s["rmse"] == pytest.approx(np.sqrt(12500))
    assert s["mape"] == pytest.approx((1.5 + 0.25 + 1 / 6 + 0.375) / 4)
    # Worse than the mean gives a negative R².
    assert mmm.scores(actual, actual[::-1])["r2"] == pytest.approx(-3.0)


def test_scores_refuse_what_they_cannot_score() -> None:
    with pytest.raises(ValueError, match="no days"):
        mmm.scores(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same length"):
        mmm.scores(np.array([1.0, 2.0]), np.array([1.0]))


def test_a_day_without_revenue_is_left_out_of_mape_only() -> None:
    s = mmm.scores(np.array([100.0, 0.0, 200.0]), np.array([110.0, 30.0, 180.0]))
    assert s["mape"] == pytest.approx((0.1 + 0.1) / 2) and s["mape_days"] == 2
    assert s["rmse"] == pytest.approx(np.sqrt((100 + 900 + 400) / 3)) and s["days"] == 3
    assert mmm.scores(np.zeros(3), np.ones(3))["mape"] is None


def test_r2_is_none_when_the_actual_values_do_not_vary() -> None:
    assert mmm.scores(np.array([250.0]), np.array([240.0]))["r2"] is None
    assert mmm.scores(np.full(3, 250.0), np.full(3, 240.0))["r2"] is None


def test_the_command_line_writes_valid_json_for_a_one_day_holdout(generated, tmp_path, capsys):
    out = tmp_path / "mmm.json"
    assert mmm.main(["--data", str(generated), "--holdout-days", "1", "--json", str(out)]) == 0
    assert "holdout  R2 n/a" in capsys.readouterr().out
    figures = json.loads(out.read_text(encoding="utf-8"), parse_constant=pytest.fail)
    assert figures["fit"]["holdout"]["r2"] is None


# --- comparison with the truth and the command line ------------------------------------------


def test_the_truth_comparison_reads_the_generators_record(small, generated) -> None:
    truth, caused = mmm.read_truth(generated, small.dates)
    td = small.truth_daily
    for i, ch in enumerate(small.config.channels):
        assert caused[ch.name].sum() == pytest.approx(
            td["revenue_cents"][td["component"] == i].sum() / 100, rel=1e-12
        )
        assert mmm.true_params(truth)[ch.name] == (
            ch.decay, ch.half_saturation * ch.daily_spend, ch.hill_shape
        )  # fmt: skip


def test_the_truth_comparison_splits_train_and_held_out_days(small, generated) -> None:
    truth, caused = mmm.read_truth(generated, small.dates)
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


def test_expected_new_revenue_is_expected_first_purchases_times_the_mean_order(
    small, generated
) -> None:
    truth, _ = mmm.read_truth(generated, small.dates)
    expected = mmm.expected_new_revenue(generated, small.dates, truth)
    td = small.truth_daily
    n_channels = len(small.config.channels)
    first = td["component"] <= n_channels  # each channel and the baseline, not repeats
    per_day = np.bincount(td["day"][first], weights=td["expected_conversions"][first])
    np.testing.assert_allclose(expected, per_day * small.config.aov, rtol=1e-5)
    # It is the expectation of what actually happened: the totals agree to within chance.
    actual = mmm.inputs_from_dataset(small).new_revenue
    assert actual.sum() == pytest.approx(expected.sum(), rel=0.05)
    # The mean order value comes from the run's record.
    halved = {**truth, "config": {**truth["config"], "aov": truth["config"]["aov"] / 2}}
    np.testing.assert_allclose(
        mmm.expected_new_revenue(generated, small.dates, halved), expected / 2, rtol=1e-12
    )


def test_the_command_line_fits_and_writes_its_figures(generated, tmp_path, capsys) -> None:
    out = tmp_path / "mmm.json"
    args = ["--data", str(generated), "--holdout-days", "20", "--json", str(out)]
    assert mmm.main(args) == 0
    printed = capsys.readouterr().out
    assert "100 training days, 20 held out" in printed and "baseline terms only" in printed
    figures = json.loads(out.read_text(encoding="utf-8"))
    assert figures["target"] == "new_customer"
    assert set(figures["expected_revenue"]) == {"train", "holdout"}
    assert set(figures["fit"]) == {"train", "holdout"}
    assert {r["channel"] for r in figures["channels"]} == {
        "paid_search", "paid_social", "display", "online_video"
    }  # fmt: skip
    assert figures["search"]["sse"] <= figures["search"]["grid_sse"]


def test_the_expected_revenue_is_only_scored_against_new_customer_revenue(
    generated, tmp_path
) -> None:
    out = tmp_path / "mmm.json"
    args = ["--data", str(generated), "--holdout-days", "20", "--target", "total"]
    assert mmm.main([*args, "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["expected_revenue"] is None


def test_the_command_line_reports_a_damaged_folder(files, capsys) -> None:
    (files / "truth_daily.csv").unlink()
    assert mmm.main(["--data", str(files)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_the_command_line_reports_a_truth_without_a_channel(files, capsys) -> None:
    truth = json.loads((files / "truth.json").read_text(encoding="utf-8"))
    del truth["channels"]["display"]
    (files / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    assert mmm.main(["--data", str(files), "--holdout-days", "20"]) == 1
    assert "no record of display" in capsys.readouterr().err


def test_the_command_line_refuses_a_holdout_longer_than_the_data(generated, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        mmm.main(["--data", str(generated), "--holdout-days", "500"])
    assert exc.value.code == 2
    assert "too few" in capsys.readouterr().err


def test_a_lift_of_nothing_is_a_measurement_and_its_refusals_say_what_is_allowed() -> None:
    assert mmm.Lift("a", START, START + np.timedelta64(27, "D"), 0.0, 10.0).revenue == 0.0
    with pytest.raises(ValueError, match="lift revenue must be a finite number, at least 0"):
        mmm.Lift("a", START, START + np.timedelta64(27, "D"), -1.0, 10.0)
    with pytest.raises(ValueError, match="lift se must be a finite number, above 0"):
        mmm.Lift("a", START, START + np.timedelta64(27, "D"), 1.0, 0.0)


@pytest.mark.parametrize("bad", [5, 5.0, True, np.int64(5), np.datetime64("NaT", "D"), None])
def test_a_lift_date_must_be_a_date(bad) -> None:
    # A bare number would be read as days since 1970 and accepted.
    with pytest.raises(ValueError, match="lift start must be a date"):
        mmm.Lift("a", bad, START + np.timedelta64(27, "D"), 1.0, 1.0)
    with pytest.raises(ValueError, match="lift end must be a date"):
        mmm.Lift("a", START, bad, 1.0, 1.0)


def test_a_lift_takes_dates_in_the_usual_forms() -> None:
    import datetime as dt

    for start in (START, "2024-01-01", dt.date(2024, 1, 1), np.datetime64("2024-01-01T00:00")):
        lift = mmm.Lift("a", start, "2024-01-28", 1.0, 1.0)
        assert np.datetime64(lift.start, "D") == START


# --- a results file's path, checked before a run --------------------------------------------


def test_a_results_path_is_checked_before_a_run(tmp_path) -> None:
    assert "is a folder" in mmm.output_problem(tmp_path)
    nested = tmp_path / "a" / "b" / "figures.json"
    assert mmm.output_problem(nested) is None
    assert nested.parent.is_dir() and not nested.exists()  # no empty file left behind
    kept = tmp_path / "kept.json"
    kept.write_text("old figures", encoding="utf-8")
    assert mmm.output_problem(kept) is None
    assert kept.read_text(encoding="utf-8") == "old figures"


@pytest.mark.parametrize("module, heavy", [("mmm", "read_inputs"), ("identify", "run_seed"),
                                           ("optimise", "plan")])  # fmt: skip
def test_a_folder_for_the_figures_is_refused_before_any_work(
    tmp_path, monkeypatch, capsys, module, heavy
) -> None:
    import importlib

    target = importlib.import_module(f"marketing_roi.{module}")

    def never(*args, **kwargs):
        raise AssertionError("the run started")

    monkeypatch.setattr(target, heavy, never)
    with pytest.raises(SystemExit):
        target.main(
            ["--seeds", "5", "--json", str(tmp_path)]
            if module == "identify"
            else ["--json", str(tmp_path)]
        )
    assert "is a folder, not a file" in capsys.readouterr().err
