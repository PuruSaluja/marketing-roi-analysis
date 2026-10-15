"""Adstock: carryover transforms that let a day's spend keep working on later days.

An ad seen today can bring a purchase next week, so a mix model regresses purchases not
on each day's spend but on its *adstock*, a weighted sum of that day's and earlier days'
spend. Each transform here maps a daily spend series to that sum:

``geometric_adstock(spend, decay)``
    ``a[t] = spend[t] + decay * a[t - 1]``: the effect fades by the same share every
    day, so it peaks on the day of spend. One parameter, over an infinite horizon.
``weibull_adstock(spend, shape, scale, max_lag, form=...)``
    Weights over a fixed window of lags taken from a Weibull distribution, so the rate
    of fading can change with the lag. ``form="survival"`` weights lag ``l`` by
    ``exp(-(l / scale) ** shape)``, the chance an effect has not yet ended: it still
    peaks on the day of spend, and with ``shape == 1`` it is geometric with ``decay =
    exp(-1 / scale)``; a shape below 1 gives a long tail, above 1 an effect that holds
    for a while and then drops. ``form="pmf"`` weights lag ``l`` by the Weibull
    probability of ``[l, l + 1)``, so with ``shape > 1`` the effect can peak after the
    day of spend, as it can for video or display: near the distribution's mode,
    ``scale * ((shape - 1) / shape) ** (1 / shape)`` days, which has to be a day or more
    (shape 2.5 and scale 6 peak 4 days later; with scale 1 they still peak on the day).

All of them **normalise** by default: the weights sum to 1, so a steady spend of $100 a
day has an adstock of $100 a day once the window has filled (the geometric form's
window never fills, so it approaches $100), and the adstock stays in dollars per day.
Unnormalised, it is the total of the decayed spend instead. Days before the series
starts count as no spend, unless ``start="steady"`` is given to the
geometric form, which treats them as the first day's spend repeated (the generator's
convention, so a constant series has a constant adstock from day one).

``half_life``, ``mean_lag`` and ``lag_for_share`` describe a transform's carryover in
days.
"""

from __future__ import annotations

import math
import numbers
from typing import Literal

import numpy as np

Form = Literal["survival", "pmf"]


def _check_spend(spend) -> np.ndarray:
    x = np.asarray(spend, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"spend must be one series of days, got an array of shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("spend must be finite numbers, without NaN or infinity")
    return x


def _is_real(value) -> bool:
    """A real number, NumPy's included, but not a true/false."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _check_positive(name: str, value: float) -> float:
    if not _is_real(value) or not value > 0:
        raise ValueError(f"{name} must be a number above 0, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _check_decay(decay: float) -> float:
    if not _is_real(decay) or not 0 <= decay < 1:
        raise ValueError(f"decay must be at least 0 and below 1, got {decay!r}")
    return float(decay)


def _check_lag(max_lag: int) -> int:
    if (
        not isinstance(max_lag, numbers.Integral)
        or isinstance(max_lag, bool | np.bool_)
        or max_lag < 1
    ):
        raise ValueError(f"max_lag must be a whole number of days, at least 1, got {max_lag!r}")
    return int(max_lag)


def geometric_adstock(
    spend,
    decay: float,
    *,
    normalise: bool = True,
    start: Literal["zero", "steady"] = "zero",
) -> np.ndarray:
    """Geometric adstock of a daily series: ``a[t] = spend[t] + decay * a[t - 1]``.

    Normalised (the default), the result is multiplied by ``1 - decay``, the weights'
    sum over an infinite horizon. ``start`` sets the days before the series: ``"zero"``,
    no spend; ``"steady"``, the first day's spend repeated for ever.
    """
    x = _check_spend(spend)
    decay = _check_decay(decay)
    if start not in ("zero", "steady"):
        raise ValueError(f"start must be 'zero' or 'steady', got {start!r}")
    out = np.empty(len(x))
    carry = x[0] / (1 - decay) if start == "steady" and len(x) else 0.0
    for t, s in enumerate(x):
        carry = s + decay * carry
        out[t] = carry
    return out * (1 - decay) if normalise else out


def geometric_weights(decay: float, max_lag: int) -> np.ndarray:
    """The geometric adstock's weights for lags 0 to ``max_lag - 1``: ``decay ** lag``."""
    decay, max_lag = _check_decay(decay), _check_lag(max_lag)
    return decay ** np.arange(max_lag, dtype=float)


def weibull_weights(
    shape: float, scale: float, max_lag: int, form: Form = "survival"
) -> np.ndarray:
    """Weights for lags 0 to ``max_lag - 1`` from a Weibull(shape, scale) in days.

    ``"survival"``: ``exp(-(lag / scale) ** shape)``, 1 at lag 0 and falling.
    ``"pmf"``: ``F(lag + 1) - F(lag)``, the probability of the day starting at ``lag``,
    where ``F`` is the Weibull distribution function.
    """
    shape, scale = _check_positive("shape", shape), _check_positive("scale", scale)
    max_lag = _check_lag(max_lag)
    lags = np.arange(max_lag + 1, dtype=float)
    survival = np.exp(-((lags / scale) ** shape))
    if form == "survival":
        return survival[:-1]
    if form == "pmf":
        return survival[:-1] - survival[1:]
    raise ValueError(f"form must be 'survival' or 'pmf', got {form!r}")


def convolve_adstock(spend, weights, *, normalise: bool = True) -> np.ndarray:
    """``a[t] = sum over lag of weights[lag] * spend[t - lag]``, no spend before day 0.

    Normalised (the default), the weights are first divided by their sum.
    """
    x = _check_spend(spend)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be a non-empty series of finite numbers, none negative")
    total = w.sum()
    if not total > 0:
        raise ValueError("weights must not all be 0")
    if normalise:
        w = w / total
    if len(x) == 0:
        return x
    return np.convolve(x, w)[: len(x)]


def weibull_adstock(
    spend,
    shape: float,
    scale: float,
    max_lag: int,
    *,
    form: Form = "survival",
    normalise: bool = True,
) -> np.ndarray:
    """Weibull adstock over lags 0 to ``max_lag - 1`` (see ``weibull_weights``)."""
    return convolve_adstock(
        spend, weibull_weights(shape, scale, max_lag, form), normalise=normalise
    )


def half_life(decay: float) -> float:
    """Days until a geometric effect has fallen to half its first day's: ``ln 0.5 / ln
    decay``. 0 for no carryover."""
    decay = _check_decay(decay)
    return 0.0 if decay == 0 else math.log(0.5) / math.log(decay)


def mean_lag(weights) -> float:
    """The weights' average lag in days, each lag counted by its weight."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or (w < 0).any() or not w.sum() > 0:
        raise ValueError("weights must be a non-empty series, none negative, not all 0")
    return float((np.arange(len(w)) * w).sum() / w.sum())


def lag_for_share(weights, share: float) -> int:
    """The first lag by which ``share`` of the weights' total has been delivered."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or (w < 0).any() or not w.sum() > 0:
        raise ValueError("weights must be a non-empty series, none negative, not all 0")
    if not _is_real(share) or not 0 < share <= 1:
        raise ValueError(f"share must be above 0 and at most 1, got {share!r}")
    cumulative = np.cumsum(w) / w.sum()
    # A share of 1 is reached at the last positive weight; rounding can leave the
    # cumulative sum a hair below 1 there.
    hit = np.flatnonzero(cumulative >= share - 1e-12)
    return int(hit[0])
