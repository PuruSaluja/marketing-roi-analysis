import datetime as dt
import math

import numpy as np
import pytest

from marketing_roi import generate as gen
from marketing_roi.adstock import (
    convolve_adstock,
    geometric_adstock,
    geometric_weights,
    half_life,
    lag_for_share,
    mean_lag,
    weibull_adstock,
    weibull_weights,
)
from marketing_roi.channels import CHANNELS

RNG = np.random.default_rng(7)
SPEND = RNG.lognormal(5, 0.6, 120) * (RNG.random(120) < 0.8)


def impulse(n: int = 30) -> np.ndarray:
    x = np.zeros(n)
    x[0] = 1.0
    return x


# --- geometric -------------------------------------------------------------------------------


@pytest.mark.parametrize("decay", [0.0, 0.1, 0.5, 0.9])
def test_a_single_day_s_spend_fades_by_the_decay_each_day(decay) -> None:
    raw = geometric_adstock(impulse(), decay, normalise=False)
    np.testing.assert_allclose(raw, decay ** np.arange(30), rtol=1e-13, atol=0)
    np.testing.assert_allclose(geometric_adstock(impulse(), decay), (1 - decay) * raw, rtol=1e-13)


def test_spend_keeps_contributing_on_later_days() -> None:
    a = geometric_adstock(impulse(), 0.7)
    assert a[0] == pytest.approx(0.3)
    assert (a[1:] > 0).all() and (np.diff(a) < 0).all()
    # Normalised, all of the day's spend is delivered over time.
    assert geometric_adstock(impulse(2000), 0.7).sum() == pytest.approx(1.0, rel=1e-12)


def test_no_decay_is_no_carryover() -> None:
    np.testing.assert_array_equal(geometric_adstock(SPEND, 0.0), SPEND)


def test_a_steady_start_keeps_a_constant_spend_constant() -> None:
    flat = np.full(50, 120.0)
    np.testing.assert_allclose(geometric_adstock(flat, 0.6, start="steady"), flat, rtol=1e-13)
    # From no spend before day 0, it builds towards the spend: 1 - decay^(t+1).
    zero = geometric_adstock(flat, 0.6)
    np.testing.assert_allclose(zero, 120 * (1 - 0.6 ** np.arange(1, 51)), rtol=1e-12)


@pytest.mark.parametrize("decay", [0.0, 0.1, 0.4, 0.7, 0.95])
def test_it_matches_the_generators_adstock(decay) -> None:
    np.testing.assert_allclose(
        geometric_adstock(SPEND, decay, start="steady"), gen.adstock(SPEND, decay), rtol=1e-12
    )


def test_it_reproduces_the_adstock_the_generator_recorded() -> None:
    data = gen.generate(gen.GeneratorConfig(seed=3, start=dt.date(2025, 1, 1), days=200))
    cd, td = data.channel_daily, data.truth_daily
    for i, ch in enumerate(data.config.channels):
        spend = cd["spend_cents"][cd["channel"] == i] / 100
        recorded = td["adstocked_spend"][td["component"] == i]
        np.testing.assert_allclose(
            geometric_adstock(spend, ch.decay, start="steady"), recorded, rtol=1e-12
        )


# --- Weibull ---------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 2.0, 7.0])
def test_a_survival_shape_of_one_is_geometric(scale) -> None:
    decay = math.exp(-1 / scale)
    np.testing.assert_allclose(
        weibull_weights(1.0, scale, 40), geometric_weights(decay, 40), rtol=1e-12
    )
    # With a window as long as the series, the two adstocks agree exactly.
    np.testing.assert_allclose(
        weibull_adstock(SPEND, 1.0, scale, len(SPEND), normalise=False),
        geometric_adstock(SPEND, decay, normalise=False),
        rtol=1e-10,
    )


def test_the_survival_shape_sets_how_the_effect_fades() -> None:
    long_tail = weibull_weights(0.5, 3.0, 60)
    holds = weibull_weights(3.0, 3.0, 60)
    assert long_tail[0] == holds[0] == 1.0
    assert (np.diff(long_tail) < 0).all() and (np.diff(holds) <= 0).all()
    # Shape above 1 holds early and then drops; below 1 drops fast and then lingers.
    assert holds[1] > long_tail[1]
    assert long_tail[20] > holds[20]
    assert mean_lag(long_tail) > mean_lag(holds)


def test_a_pmf_weibull_peaks_after_the_day_of_spend() -> None:
    w = weibull_weights(2.5, 6.0, 40, form="pmf")
    mode = 6.0 * ((2.5 - 1) / 2.5) ** (1 / 2.5)  # about 4.9 days
    assert int(np.argmax(w)) == math.floor(mode) == 4  # the day holding the mode
    assert np.argmax(w) > 0
    assert w.sum() == pytest.approx(1 - math.exp(-((40 / 6.0) ** 2.5)), rel=1e-12)
    a = weibull_adstock(impulse(40), 2.5, 6.0, 40, form="pmf")
    assert int(np.argmax(a)) == int(np.argmax(w))
    # Shape 1 or below has its largest mass on the first day.
    assert np.argmax(weibull_weights(1.0, 6.0, 40, form="pmf")) == 0
    assert np.argmax(weibull_weights(0.7, 6.0, 40, form="pmf")) == 0


@pytest.mark.parametrize(
    "transform",
    [
        lambda x: geometric_adstock(x, 0.6),
        lambda x: weibull_adstock(x, 0.8, 4.0, 30),
        lambda x: weibull_adstock(x, 2.0, 5.0, 30, form="pmf"),
    ],
)
def test_every_transform_is_linear_and_keeps_a_steady_spend_once_filled(transform) -> None:
    other = RNG.lognormal(4, 0.3, len(SPEND))
    np.testing.assert_allclose(
        transform(2 * SPEND + 3 * other), 2 * transform(SPEND) + 3 * transform(other), rtol=1e-10
    )
    flat = transform(np.full(400, 50.0))
    assert flat[-1] == pytest.approx(50.0, rel=1e-6)
    assert (flat <= 50.0 + 1e-9).all()


def test_convolution_reads_only_earlier_days() -> None:
    x = SPEND.copy()
    a = convolve_adstock(x, [0.5, 0.3, 0.2])
    x[60:] = 0
    np.testing.assert_array_equal(convolve_adstock(x, [0.5, 0.3, 0.2])[:60], a[:60])
    np.testing.assert_allclose(a[2], 0.5 * SPEND[2] + 0.3 * SPEND[1] + 0.2 * SPEND[0])
    raw = convolve_adstock(SPEND, [2.0, 1.0], normalise=False)
    assert raw[1] == pytest.approx(2 * SPEND[1] + SPEND[0])


def test_an_empty_series_gives_an_empty_adstock() -> None:
    assert len(geometric_adstock([], 0.5)) == 0
    assert len(geometric_adstock([], 0.5, start="steady")) == 0
    assert len(weibull_adstock([], 1.0, 2.0, 5)) == 0


# --- carryover in days -----------------------------------------------------------------------


@pytest.mark.parametrize("decay", [0.1, 0.5, 0.7, 0.99])
def test_half_life_is_where_the_effect_halves(decay) -> None:
    assert decay ** half_life(decay) == pytest.approx(0.5, rel=1e-12)
    assert half_life(0.0) == 0.0


def test_mean_lag_of_geometric_weights() -> None:
    assert mean_lag(geometric_weights(0.7, 2000)) == pytest.approx(0.7 / 0.3, rel=1e-10)
    assert mean_lag([1.0]) == 0.0


def test_lag_for_share() -> None:
    w = geometric_weights(0.5, 60)  # shares 1/2, 1/4, ...
    assert lag_for_share(w, 0.5) == 0
    assert lag_for_share(w, 0.75) == 1
    assert lag_for_share(w, 0.9) == 3
    assert lag_for_share([0.0, 0.0, 3.0, 0.0], 1) == 2
    assert lag_for_share([1.0, 2.0, 1.0], 1) == 2


def test_the_generators_channels_carry_over_as_documented() -> None:
    # The table in the README: the true decays' carryover in days.
    got = {
        c.name: (round(half_life(c.decay), 2), lag_for_share(geometric_weights(c.decay, 400), 0.9))
        for c in CHANNELS
    }
    assert got == {
        "paid_search": (0.3, 0),
        "paid_social": (0.76, 2),
        "display": (1.0, 3),
        "online_video": (1.94, 6),
    }


# --- refusals --------------------------------------------------------------------------------


@pytest.mark.parametrize("decay", [1, 1.0, -0.1, float("nan"), True, "0.5", None])
def test_a_bad_decay_is_refused(decay) -> None:
    with pytest.raises(ValueError, match="decay must be at least 0 and below 1"):
        geometric_adstock(SPEND, decay)
    with pytest.raises(ValueError, match="decay must be"):
        half_life(decay)


@pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan"), True, "2"])
def test_a_bad_shape_or_scale_is_refused(value) -> None:
    with pytest.raises(ValueError, match="shape must be"):
        weibull_weights(value, 2.0, 10)
    with pytest.raises(ValueError, match="scale must be"):
        weibull_weights(1.0, value, 10)


@pytest.mark.parametrize("max_lag", [0, -3, 1.5, True, "7"])
def test_a_bad_window_is_refused(max_lag) -> None:
    with pytest.raises(ValueError, match="max_lag must be a whole number"):
        weibull_adstock(SPEND, 1.0, 2.0, max_lag)
    with pytest.raises(ValueError, match="max_lag must be a whole number"):
        geometric_weights(0.5, max_lag)


@pytest.mark.parametrize("spend", [np.ones((2, 5)), [1.0, float("nan")], [float("inf")], [[1.0]]])
def test_bad_spend_is_refused(spend) -> None:
    with pytest.raises(ValueError, match="spend must be"):
        geometric_adstock(spend, 0.5)
    with pytest.raises(ValueError, match="spend must be"):
        convolve_adstock(spend, [1.0])


@pytest.mark.parametrize("weights", [[], [1.0, -0.5], [0.0, 0.0], [float("nan")], [[1.0]]])
def test_bad_weights_are_refused(weights) -> None:
    with pytest.raises(ValueError, match="weights must"):
        convolve_adstock(SPEND, weights)


def test_bad_options_are_refused() -> None:
    with pytest.raises(ValueError, match="form must be 'survival' or 'pmf'"):
        weibull_weights(1.0, 2.0, 10, form="cdf")
    with pytest.raises(ValueError, match="start must be 'zero' or 'steady'"):
        geometric_adstock(SPEND, 0.5, start="first")
    for share in (0, 1.5, -0.2, True):
        with pytest.raises(ValueError, match="share must be above 0"):
            lag_for_share([1.0], share)
    for weights in ([], [0.0], [-1.0, 2.0]):
        with pytest.raises(ValueError, match="weights must"):
            mean_lag(weights)
        with pytest.raises(ValueError, match="weights must"):
            lag_for_share(weights, 0.5)
