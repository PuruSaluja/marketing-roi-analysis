"""Saturation: response curves that flatten as a channel's spend grows.

The first dollars a channel spends reach the people most likely to buy; later dollars
reach fewer new ones, so each extra dollar brings less. A mix model captures that by
passing each channel's adstocked spend ``x`` (dollars per day, see ``adstock``) through
a curve that rises ever more slowly. Two are here:

``hill(x, half_saturation, shape)``
    ``x ** shape / (x ** shape + half_saturation ** shape)``, rising from 0 towards a
    ceiling of 1, which it reaches half of at ``half_saturation``. With ``shape`` above
    1 it is an S: the first dollars do little, the middle most, the last little again.
    With ``shape`` at or below 1 every extra dollar does less than the one before.
    This is the curve the generator uses.
``log_response(x, scale)``
    ``ln(1 + x / scale)``: no ceiling, but each extra dollar still does less, its
    effect falling as ``1 / (scale + x)``.

A channel's curve, ``ResponseCurve``, multiplies one of them by ``beta``, so it gives
expected purchases (or revenue) a day at a given adstocked spend. At a steady spend the
adstock equals the spend (the transforms normalise), so the curve is also the steady
daily response to a daily budget. Its ``marginal`` is the derivative, the response to
one more dollar a day, and its inverse, ``marginal_cost``, the cost of one more purchase
at that spend: the figure a budget reallocation has to equalise across channels.

``true_curves`` builds each channel's curve as the generator made it, the truth a fitted
model is checked against.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Literal

import numpy as np

from marketing_roi.channels import Channel

Kind = Literal["hill", "log"]


def _is_real(value) -> bool:
    """A real number, NumPy's included, but not a true/false."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _check_positive(name: str, value) -> float:
    if not _is_real(value) or not value > 0:
        raise ValueError(f"{name} must be a number above 0, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _check_spend(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("spend must be finite numbers, without NaN or infinity")
    if (a < 0).any():
        raise ValueError("spend must not be negative")
    return a


def _out(a: np.ndarray, x) -> np.ndarray | float:
    """A float for a scalar input, an array otherwise."""
    return float(a) if np.ndim(x) == 0 else a


def hill(x, half_saturation: float, shape: float) -> np.ndarray | float:
    """``x ** shape / (x ** shape + half_saturation ** shape)``: 0 at 0, 1/2 at
    ``half_saturation``, approaching 1.

    Computed as ``1 / (1 + (half_saturation / x) ** shape)``, which stays finite where
    ``x ** shape`` alone would overflow.
    """
    a = _check_spend(x)
    k = _check_positive("half_saturation", half_saturation)
    s = _check_positive("shape", shape)
    with np.errstate(divide="ignore", over="ignore"):
        ratio = np.power(k / a, s)
    return _out(np.where(a > 0, 1.0 / (1.0 + ratio), 0.0), x)


def hill_marginal(x, half_saturation: float, shape: float) -> np.ndarray | float:
    """The Hill curve's slope: ``shape / x * h * (1 - h)`` where ``h = hill(x)``.

    At 0 it is infinite for a shape below 1, ``1 / half_saturation`` for shape 1, and
    0 above.
    """
    a = _check_spend(x)
    k = _check_positive("half_saturation", half_saturation)
    s = _check_positive("shape", shape)
    h = np.asarray(hill(a, k, s), dtype=float)
    at_zero = math.inf if s < 1 else (1.0 / k if s == 1 else 0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        slope = s / a * h * (1.0 - h)
    # A spend so small that s / a overflows while h underflows gives inf * 0; the slope
    # there is the limit at 0.
    return _out(np.where((a > 0) & ~np.isnan(slope), slope, at_zero), x)


def hill_inverse(share, half_saturation: float, shape: float) -> np.ndarray | float:
    """The spend at which the Hill curve reaches ``share`` of its ceiling, ``share`` in
    [0, 1): ``half_saturation * (share / (1 - share)) ** (1 / shape)``."""
    y = np.asarray(share, dtype=float)
    if not np.isfinite(y).all() or (y < 0).any() or (y >= 1).any():
        raise ValueError("share must be at least 0 and below 1")
    k = _check_positive("half_saturation", half_saturation)
    s = _check_positive("shape", shape)
    return _out(k * np.power(y / (1.0 - y), 1.0 / s), share)


def log_response(x, scale: float) -> np.ndarray | float:
    """``ln(1 + x / scale)``: 0 at 0, ``ln 2`` at ``scale``, without a ceiling."""
    a = _check_spend(x)
    c = _check_positive("scale", scale)
    return _out(np.log1p(a / c), x)


def log_marginal(x, scale: float) -> np.ndarray | float:
    """The log curve's slope: ``1 / (scale + x)``."""
    a = _check_spend(x)
    c = _check_positive("scale", scale)
    return _out(1.0 / (c + a), x)


@dataclass(frozen=True)
class ResponseCurve:
    """``beta`` times a Hill or log curve of a channel's adstocked daily spend.

    ``half_saturation`` and ``shape`` are the Hill curve's; ``scale`` the log curve's.
    """

    kind: Kind
    beta: float
    half_saturation: float | None = None
    shape: float | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        if not _is_real(self.beta) or not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError(f"beta must be a finite number, at least 0, got {self.beta!r}")
        if self.kind == "hill":
            _check_positive("half_saturation", self.half_saturation)
            _check_positive("shape", self.shape)
            if self.scale is not None:
                raise ValueError("a Hill curve takes no scale")
        elif self.kind == "log":
            _check_positive("scale", self.scale)
            if self.half_saturation is not None or self.shape is not None:
                raise ValueError("a log curve takes no half_saturation or shape")
        else:
            raise ValueError(f"kind must be 'hill' or 'log', got {self.kind!r}")

    def response(self, x) -> np.ndarray | float:
        """Expected response a day at adstocked daily spend ``x``."""
        if self.kind == "hill":
            curve = hill(x, self.half_saturation, self.shape)
        else:
            curve = log_response(x, self.scale)
        return self.beta * curve

    def marginal(self, x) -> np.ndarray | float:
        """Extra response a day from one more dollar a day at ``x``."""
        if self.kind == "hill":
            slope = hill_marginal(x, self.half_saturation, self.shape)
        else:
            slope = log_marginal(x, self.scale)
        return self.beta * slope

    def average_cost(self, x) -> np.ndarray | float:
        """Spend per unit of response at ``x``: ``x / response(x)``; infinite where the
        response is 0."""
        a = _check_spend(x)
        r = np.asarray(self.response(a), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = np.where(r > 0, a / r, math.inf)
        return _out(cost, x)

    def marginal_cost(self, x) -> np.ndarray | float:
        """Spend per extra unit of response from one more dollar at ``x``: ``1 /
        marginal(x)``; infinite where the slope is 0."""
        m = np.asarray(self.marginal(x), dtype=float)
        with np.errstate(divide="ignore"):
            cost = np.where(m > 0, 1.0 / np.where(m > 0, m, 1.0), math.inf)
        return _out(cost, x)


def true_curve(channel: Channel, beta: float) -> ResponseCurve:
    """The generator's response for ``channel``: expected purchases a day at an
    adstocked spend, ``beta * hill(x, half_saturation * daily_spend, hill_shape)``."""
    return ResponseCurve(
        kind="hill",
        beta=beta,
        half_saturation=channel.half_saturation * channel.daily_spend,
        shape=channel.hill_shape,
    )


def true_curves(channels, betas: dict[str, float]) -> dict[str, ResponseCurve]:
    """Each channel's true purchase curve, from a run's betas (``Dataset.betas``, or the
    ``beta`` recorded per channel in truth.json)."""
    return {ch.name: true_curve(ch, betas[ch.name]) for ch in channels}
