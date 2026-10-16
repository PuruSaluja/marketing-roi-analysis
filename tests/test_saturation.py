import datetime as dt
import math
import warnings

import numpy as np
import pytest

from marketing_roi import generate as gen
from marketing_roi.channels import CHANNELS
from marketing_roi.saturation import (
    ResponseCurve,
    hill,
    hill_inverse,
    hill_marginal,
    log_marginal,
    log_response,
    true_curve,
    true_curves,
)

X = np.array([0.0, 1e-3, 0.5, 10.0, 250.0, 900.0, 4000.0, 1e6])


def numeric_slope(f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Central difference, relative step."""
    step = h * np.maximum(x, 1.0)
    return (f(x + step) - f(x - step)) / (2 * step)


# --- Hill ------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [0.5, 1.0, 1.3, 3.0])
def test_hill_is_half_at_the_half_saturation_point(shape) -> None:
    assert hill(0.0, 720.0, shape) == 0.0
    assert hill(720.0, 720.0, shape) == pytest.approx(0.5, rel=1e-15)
    assert hill(1e20, 720.0, shape) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize(("k", "s"), [(720.0, 1.2), (720.0, 1.5), (300.0, 1.3), (800.0, 1.0)])
def test_hill_matches_the_generators_curve(k, s) -> None:
    np.testing.assert_allclose(hill(X, k, s), gen.hill(X, k, s), rtol=1e-13, atol=0)


def test_hill_rises_and_never_passes_its_ceiling() -> None:
    x = np.linspace(0, 1e5, 5001)
    for s in (0.5, 1.0, 2.5):
        y = hill(x, 900.0, s)
        assert (np.diff(y) > 0).all()
        assert ((y >= 0) & (y < 1)).all()


def test_hill_stays_finite_where_the_power_overflows() -> None:
    # x ** shape overflows to infinity, and the generator's form gives inf / inf = NaN.
    with np.errstate(over="ignore", invalid="ignore"):
        assert math.isnan(gen.hill(1e300, 1.0, 3.0))
    assert hill(1e300, 1.0, 3.0) == 1.0
    assert hill(1e-300, 1.0, 3.0) == 0.0


@pytest.mark.parametrize("shape", [0.7, 1.0, 1.3, 2.5])
def test_hill_marginal_is_the_slope(shape) -> None:
    x = X[X > 0.1]
    np.testing.assert_allclose(
        hill_marginal(x, 900.0, shape),
        numeric_slope(lambda v: hill(v, 900.0, shape), x),
        rtol=1e-6,
        atol=1e-15,
    )


def test_hill_marginal_at_zero_follows_the_shape() -> None:
    assert hill_marginal(0.0, 900.0, 0.8) == math.inf
    assert hill_marginal(0.0, 900.0, 1.0) == pytest.approx(1 / 900)
    assert hill_marginal(0.0, 900.0, 1.4) == 0.0


def test_an_s_shaped_hill_has_its_steepest_point_after_zero() -> None:
    # Shape s > 1: the slope peaks at k * ((s - 1) / (s + 1)) ** (1 / s).
    k, s = 900.0, 2.0
    x = np.linspace(1, 3000, 30000)
    peak = x[np.argmax(hill_marginal(x, k, s))]
    assert peak == pytest.approx(k * ((s - 1) / (s + 1)) ** (1 / s), abs=0.2)


@pytest.mark.parametrize("shape", [0.6, 1.0, 1.5])
def test_hill_inverse_undoes_hill(shape) -> None:
    shares = np.array([0.0, 0.1, 0.5, 0.9, 0.999])
    x = hill_inverse(shares, 640.0, shape)
    np.testing.assert_allclose(hill(x, 640.0, shape), shares, rtol=1e-12, atol=1e-15)
    assert hill_inverse(0.5, 640.0, shape) == pytest.approx(640.0)


@pytest.mark.parametrize("share", [-0.1, 1.0, 1.5, math.nan])
def test_hill_inverse_refuses_a_share_outside_0_to_1(share) -> None:
    with pytest.raises(ValueError, match="share"):
        hill_inverse(share, 640.0, 1.2)


# --- log -------------------------------------------------------------------------------------


def test_log_response_has_no_ceiling_but_diminishing_returns() -> None:
    assert log_response(0.0, 500.0) == 0.0
    assert log_response(500.0, 500.0) == pytest.approx(math.log(2))
    x = np.linspace(0, 1e6, 1001)
    y = log_response(x, 500.0)
    assert (np.diff(y) > 0).all() and (np.diff(y, 2) < 0).all()
    assert y[-1] > 7


def test_log_marginal_is_the_slope() -> None:
    x = X[X > 0.1]
    np.testing.assert_allclose(
        log_marginal(x, 500.0), numeric_slope(lambda v: log_response(v, 500.0), x), rtol=1e-6
    )


# --- validation ------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, math.nan, True, "1", None])
def test_parameters_must_be_positive_finite_numbers(bad) -> None:
    with pytest.raises(ValueError, match="half_saturation"):
        hill(X, bad, 1.0)
    with pytest.raises(ValueError, match="shape"):
        hill_marginal(X, 900.0, bad)
    with pytest.raises(ValueError, match="scale"):
        log_response(X, bad)


def test_numpy_numbers_are_taken_as_parameters() -> None:
    grid = np.arange(1, 4)  # np.int64, as a grid search gives
    assert hill(900.0, grid[0] * 900, np.float32(1.5)) == pytest.approx(0.5)
    assert log_response(500.0, np.int64(500)) == pytest.approx(math.log(2))
    with pytest.raises(ValueError, match="shape"):
        hill(900.0, 900.0, np.True_)


@pytest.mark.parametrize("bad", [[-1.0], [math.nan], [math.inf]])
def test_spend_must_be_finite_and_not_negative(bad) -> None:
    for f in (lambda x: hill(x, 1.0, 1.0), lambda x: log_marginal(x, 1.0)):
        with pytest.raises(ValueError, match="spend"):
            f(bad)


def test_a_scalar_gives_a_float_and_an_array_an_array() -> None:
    assert type(hill(5.0, 1.0, 1.0)) is float
    assert type(log_marginal(np.float64(5.0), 1.0)) is float
    assert hill([5.0], 1.0, 1.0).shape == (1,)
    assert hill(np.zeros((2, 3)), 1.0, 1.0).shape == (2, 3)


# --- curves ----------------------------------------------------------------------------------


def test_a_curve_is_beta_times_its_shape() -> None:
    c = ResponseCurve("hill", beta=40.0, half_saturation=720.0, shape=1.2)
    np.testing.assert_allclose(c.response(X), 40 * hill(X, 720.0, 1.2), rtol=1e-15)
    np.testing.assert_allclose(c.marginal(X[1:]), 40 * hill_marginal(X[1:], 720.0, 1.2))
    lg = ResponseCurve("log", beta=12.0, scale=300.0)
    np.testing.assert_allclose(lg.response(X), 12 * np.log1p(X / 300), rtol=1e-15)
    np.testing.assert_allclose(lg.marginal(X), 12 / (300 + X), rtol=1e-15)


def test_costs_are_spend_over_response_and_one_over_the_slope() -> None:
    c = ResponseCurve("hill", beta=40.0, half_saturation=720.0, shape=1.2)
    assert c.average_cost(720.0) == pytest.approx(720 / 20)
    assert c.marginal_cost(500.0) == pytest.approx(1 / c.marginal(500.0))
    # Diminishing returns past the S's steep part: the next purchase costs more than
    # the average one.
    assert c.marginal_cost(900.0) > c.average_cost(900.0)
    assert c.average_cost(0.0) == math.inf
    assert c.marginal_cost(0.0) == math.inf  # shape above 1: flat at zero
    assert ResponseCurve("hill", 40.0, 720.0, 0.8).marginal_cost(0.0) == 0.0
    assert ResponseCurve("hill", 0.0, 720.0, 1.2).marginal_cost(900.0) == math.inf


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "hill", "beta": 1.0, "shape": 1.0}, "half_saturation"),
        ({"kind": "hill", "beta": 1.0, "half_saturation": 1.0}, "shape"),
        ({"kind": "hill", "beta": 1.0, "half_saturation": 1.0, "shape": 1.0, "scale": 1.0}, "no"),
        ({"kind": "log", "beta": 1.0}, "scale"),
        ({"kind": "log", "beta": 1.0, "scale": 1.0, "shape": 1.0}, "no"),
        ({"kind": "power", "beta": 1.0}, "kind"),
        ({"kind": "log", "beta": -1.0, "scale": 1.0}, "beta"),
        ({"kind": "log", "beta": math.nan, "scale": 1.0}, "beta"),
        ({"kind": "log", "beta": True, "scale": 1.0}, "beta"),
    ],
)
def test_a_curve_refuses_parameters_that_do_not_fit_its_kind(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        ResponseCurve(**kwargs)


def test_true_curves_reproduce_the_generators_expected_purchases() -> None:
    data = gen.generate(gen.GeneratorConfig(seed=3, start=dt.date(2025, 1, 1), days=200))
    curves = true_curves(data.config.channels, data.betas)
    td = data.truth_daily
    for i, ch in enumerate(data.config.channels):
        rows = td["component"] == i
        np.testing.assert_allclose(
            curves[ch.name].response(td["adstocked_spend"][rows]),
            td["expected_conversions"][rows],
            rtol=1e-12,
        )


def test_a_true_curve_is_half_way_at_its_channels_half_saturation_spend() -> None:
    for ch in CHANNELS:
        c = true_curve(ch, 10.0)
        assert c.half_saturation == ch.half_saturation * ch.daily_spend
        assert c.response(ch.half_saturation * ch.daily_spend) == pytest.approx(5.0)


@pytest.mark.parametrize("spend", [1e-310, 5e-324])
def test_hill_marginal_at_a_vanishing_spend_is_its_limit_at_zero(spend) -> None:
    # s / x overflows while the curve underflows to 0, which would give inf * 0 = NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hill_marginal(spend, 900.0, 0.7) == math.inf
        assert hill_marginal(spend, 900.0, 1.0) == pytest.approx(1 / 900)
        assert hill_marginal(spend, 900.0, 1.5) == 0.0
