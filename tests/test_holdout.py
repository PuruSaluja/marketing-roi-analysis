import numpy as np
import pytest

from marketing_roi import holdout, mmm
from marketing_roi.generate import GeneratorConfig, generate
from marketing_roi.saturation import ResponseCurve, hill, true_curves

CHANNELS = ("a", "b")
DAY0 = np.datetime64("2025-01-01", "D")


def spend_of(days: int = 30) -> np.ndarray:
    t = np.arange(days, dtype=float)
    return np.vstack([100 + 10 * np.sin(t), np.where(t % 7 < 4, 50.0, 0.0)])


# --- reallocated ------------------------------------------------------------------------------


def test_a_split_scales_each_channel_and_keeps_the_held_out_total() -> None:
    spend = spend_of()
    current = {"a": 100.0, "b": 25.0}
    split = {"a": 80.0, "b": 45.0}
    got = holdout.reallocated(spend, 20, CHANNELS, current, split)
    assert np.array_equal(got[:, :20], spend[:, :20])
    assert got[:, 20:].sum() == pytest.approx(spend[:, 20:].sum(), rel=1e-12)
    # Each channel by its own factor, then both by one common factor.
    ratio = got[:, 20:] / np.where(spend[:, 20:] > 0, spend[:, 20:], np.nan)
    a, b = np.nanmax(ratio[0]), np.nanmax(ratio[1])
    assert np.nanmin(ratio[0]) == pytest.approx(a) and np.nanmin(ratio[1]) == pytest.approx(b)
    assert b / a == pytest.approx((45 / 25) / (80 / 100))
    assert (got[1, 20:][spend[1, 20:] == 0] == 0).all()  # dark days stay dark


def test_the_current_split_changes_nothing() -> None:
    spend = spend_of()
    current = {"a": 100.0, "b": 25.0}
    got = holdout.reallocated(spend, 10, CHANNELS, current, dict(current))
    assert np.allclose(got, spend, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "start, current, split, match",
    [
        (30, {"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0}, "start"),
        (-1, {"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0}, "start"),
        (10, {"a": 1.0}, {"a": 1.0, "b": 1.0}, "name the spend's channels"),
        (10, {"a": 1.0, "b": 1.0}, {"a": 1.0, "c": 1.0}, "name the spend's channels"),
        (10, {"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 1.0}, "above 0"),
        (10, {"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0}, "no spend"),
    ],
)  # fmt: skip
def test_a_split_that_cannot_be_applied_is_refused(start, current, split, match) -> None:
    with pytest.raises(ValueError, match=match):
        holdout.reallocated(spend_of(), start, CHANNELS, current, split)


# --- the truth's revenue ----------------------------------------------------------------------


def test_the_true_revenue_is_the_generator_s_expected_purchases_times_the_order_value() -> None:
    data = generate(GeneratorConfig(seed=7, days=120))
    inputs = mmm.inputs_from_dataset(data)
    curves = {
        c: ResponseCurve("hill", p.beta * data.config.aov, p.half_saturation, p.shape)
        for c, p in true_curves(data.config.channels, data.betas).items()
    }
    decays = {ch.name: ch.decay for ch in data.config.channels}
    got = holdout.true_media_revenue(curves, decays, inputs.spend)
    td = data.truth_daily
    paid = td["component"] < len(data.config.channels)
    expected = np.bincount(td["day"][paid], weights=td["expected_conversions"][paid], minlength=120)
    assert got == pytest.approx(expected * data.config.aov, rel=1e-12)


# --- validate ---------------------------------------------------------------------------------


def model_of(betas=(400.0, 300.0), decays=(0.5, 0.2), coef_days: int = 60) -> mmm.Fit:
    _, names = mmm.baseline_design(DAY0 + np.arange(2))
    coef = np.zeros(len(names))
    coef[0] = 1000.0  # intercept only: it cancels in every lift
    channels = [
        mmm.ChannelFit("a", decays[0], 100.0, 1.5, betas[0]),
        mmm.ChannelFit("b", decays[1], 30.0, 1.0, betas[1]),
    ]
    return mmm.Fit(channels, names, coef, coef_days - 14, 14, 2, "new_customer", 0.0, 0, 0.0)


def inputs_of(days: int = 60) -> mmm.Inputs:
    dates = DAY0 + np.arange(days)
    spend = spend_of(days)
    revenue = np.full(days, 2000.0) + np.arange(days)
    return mmm.Inputs(dates, CHANNELS, spend, revenue, revenue * 0.6)


def by_hand_fitted(model: mmm.Fit, spend: np.ndarray) -> np.ndarray:
    out = np.zeros(spend.shape[1])
    for c, ch in enumerate(model.channels):
        a, carry = np.empty(spend.shape[1]), 0.0  # the model's start: no spend before
        for t, x in enumerate(spend[c]):
            carry = (1 - ch.decay) * x + ch.decay * carry
            a[t] = carry
        out += ch.beta * hill(a, ch.half_saturation, ch.shape)
    return out


def by_hand_true(curves, decays, spend: np.ndarray) -> np.ndarray:
    out = np.zeros(spend.shape[1])
    for c, (name, curve) in enumerate(curves.items()):
        a, carry = np.empty(spend.shape[1]), spend[c, 0]  # the generator's: as if steady
        for t, x in enumerate(spend[c]):
            carry = (1 - decays[name]) * x + decays[name] * carry
            a[t] = carry
        out += curve.beta * hill(a, curve.half_saturation, curve.shape)
    return out


def test_each_split_s_projected_and_true_lift_are_over_the_held_out_days() -> None:
    inputs = inputs_of()
    models = {"m1": model_of(), "m2": model_of(betas=(100.0, 900.0), decays=(0.0, 0.7))}
    truth = {"a": ResponseCurve("hill", 500.0, 80.0, 1.2), "b": ResponseCurve("hill", 200, 20, 1)}
    decays = {"a": 0.3, "b": 0.6}
    current = {"a": 100.0, "b": 28.0}
    splits = {"up_b": {"a": 70.0, "b": 58.0}, "same": dict(current)}
    got = holdout.validate(inputs, models, truth, decays, current, splits, 14)
    assert got["days"] == 14 and got["first_day"] == "2025-02-16"
    assert got["spend"] == pytest.approx(inputs.spend[:, 46:].sum())
    assert got["new_customer_revenue"] == pytest.approx(inputs.new_revenue[46:].sum())

    new = holdout.reallocated(inputs.spend, 46, CHANNELS, current, splits["up_b"])
    s = got["splits"]["up_b"]
    for name, m in models.items():
        lift = (by_hand_fitted(m, new) - by_hand_fitted(m, inputs.spend))[46:].sum()
        assert s["projected"][name] == pytest.approx(lift, rel=1e-9)
        assert s["projected_share"][name] == pytest.approx(lift / got["new_customer_revenue"])
        assert s["projected_roi"][name] == pytest.approx(lift / got["spend"])
    true = (by_hand_true(truth, decays, new) - by_hand_true(truth, decays, inputs.spend))[46:]
    assert s["true"] == pytest.approx(true.sum(), rel=1e-9)
    assert s["true_roi"] == pytest.approx(true.sum() / got["spend"])
    assert s["true_share"] == pytest.approx(true.sum() / got["new_customer_revenue"])
    assert sum(s["spend"].values()) == pytest.approx(got["spend"], rel=1e-12)
    assert s["true"] != 0 and s["projected"]["m1"] != s["projected"]["m2"]

    same = got["splits"]["same"]
    assert same["true"] == pytest.approx(0, abs=1e-6)
    assert all(v == pytest.approx(0, abs=1e-6) for v in same["projected"].values())


def test_the_forecast_error_is_each_fit_s_held_out_total_at_the_spend_as_it_was() -> None:
    inputs = inputs_of()
    m = model_of()
    got = holdout.validate(inputs, {"m": m}, {"a": ResponseCurve("hill", 1, 1, 1),
                                              "b": ResponseCurve("hill", 1, 1, 1)},
                           {"a": 0.0, "b": 0.0}, {"a": 100.0, "b": 28.0}, {}, 14)  # fmt: skip
    predicted = (1000.0 + by_hand_fitted(m, inputs.spend))[46:].sum()
    assert got["forecast"]["m"]["predicted"] == pytest.approx(predicted, rel=1e-12)
    actual = inputs.new_revenue[46:].sum()
    assert got["forecast"]["m"]["error_share"] == pytest.approx(predicted / actual - 1)


def test_validation_needs_training_days_and_the_channels_in_order() -> None:
    inputs = inputs_of()
    truth = {"a": ResponseCurve("hill", 1, 1, 1), "b": ResponseCurve("hill", 1, 1, 1)}
    args = ({"m": model_of()}, truth, {"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}, {})
    for days in (0, 60, 61):
        with pytest.raises(ValueError, match="training days"):
            holdout.validate(inputs, *args, days)
    backwards = {"b": truth["b"], "a": truth["a"]}
    with pytest.raises(ValueError, match="channel order"):
        holdout.validate(inputs, args[0], backwards, *args[2:], 14)
