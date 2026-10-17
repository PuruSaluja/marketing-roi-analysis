"""Marketing mix model: daily revenue regressed on adstocked, saturated spend.

The model reads only what a business would see, each paid channel's daily spend and
the shop's daily revenue from GA4, and explains revenue on day ``t`` as

    baseline(t) + sum over channels of beta_c * hill(adstock_c(spend_c)[t], k_c, s_c)

where the baseline is an intercept, a linear trend, yearly seasonality (sine and cosine
pairs), a weekday pattern and a Black Friday to Cyber Monday window. ``adstock_c`` is the
normalised geometric carryover with decay ``d_c`` (days before the data count as no
spend) and ``hill`` the saturation curve with half-saturation ``k_c`` and shape ``s_c``.

Given the curves' parameters ``(d_c, k_c, s_c)`` the model is linear, so the betas and
baseline come from least squares, with every beta held at 0 or above (a channel cannot
lose sales by spending). The curves' parameters are searched for: a pass over a coarse
grid one channel at a time, then a Nelder-Mead search over all of them at once, within
bounds (decay 0 to 0.95, half-saturation 0.1 to 10 times the channel's mean daily
spend, shape 0.5 to 4). Both minimise the squared error on the **training days only**.

The revenue modelled is by default **new-customer revenue**: each GA4 user id's first
purchase. Paid media in the generator causes first purchases, and repeat purchases follow
from earlier customers whatever brought them, so new-customer revenue is what spend
moves directly. It also leaves out a start-up artefact of the generator, which begins
with no customers, so its repeat purchases climb for most of the first year, a climb a
linear trend cannot follow. ``target="total"`` models all revenue instead.

``fit`` holds out the last ``holdout_days`` days. ``evaluate`` reports R², MAPE and RMSE
on the training days and, separately, on the held-out ones, which the fit never saw:
the held-out figures are the model's real accuracy. Spend is known on every day, so
the held-out days' adstock carries over the spend before them, as it would in a
forecast made with a planned budget.

``python -m marketing_roi.mmm`` fits the generated data and prints the fit, the
held-out accuracy and, because the generator recorded it, how close each channel's
fitted curve and revenue come to the truth. The fit itself never reads the truth.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import numbers
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from marketing_roi.adstock import geometric_adstock
from marketing_roi.generate import black_friday
from marketing_roi.saturation import ResponseCurve, hill

DECAY_BOUNDS = (0.0, 0.95)
HALF_SATURATION_BOUNDS = (0.1, 10.0)
"""As multiples of the channel's mean daily spend over the training days."""
SHAPE_BOUNDS = (0.5, 4.0)

GRID_DECAY = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)
GRID_HALF_SATURATION = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GRID_SHAPE = (0.6, 1.0, 1.6, 2.5)

Target = Literal["new_customer", "total"]

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat")
"""Sunday is the reference day."""


@dataclass(frozen=True)
class Inputs:
    """One row per day: each paid channel's spend, the shop's revenue, and the part of it
    from new customers (each user id's first purchase)."""

    dates: np.ndarray
    """datetime64[D], consecutive days."""
    channels: tuple[str, ...]
    spend: np.ndarray
    """Dollars, shape (channels, days)."""
    revenue: np.ndarray
    """Dollars, shape (days,)."""
    new_revenue: np.ndarray
    """Dollars from first purchases, shape (days,)."""

    def __post_init__(self) -> None:
        n = len(self.dates)
        if n < 2 or not (np.diff(self.dates.astype("datetime64[D]")).astype(int) == 1).all():
            raise ValueError("dates must be at least two consecutive days")
        if self.spend.shape != (len(self.channels), n) or not (
            self.revenue.shape == self.new_revenue.shape == (n,)
        ):
            raise ValueError("spend must be (channels, days) and revenue (days,)")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must be distinct, at least one")
        for name, a in (
            ("spend", self.spend),
            ("revenue", self.revenue),
            ("new_revenue", self.new_revenue),
        ):
            if not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f"{name} must be finite and not negative")
        if (self.new_revenue > self.revenue + 1e-6).any():
            raise ValueError("new_revenue cannot exceed revenue")

    def __len__(self) -> int:
        return len(self.dates)

    def target(self, kind: Target) -> np.ndarray:
        if kind == "new_customer":
            return self.new_revenue
        if kind == "total":
            return self.revenue
        raise ValueError(f"target must be 'new_customer' or 'total', got {kind!r}")


def inputs_from_dataset(data) -> Inputs:
    """The model's inputs from a generated ``Dataset``: spend per channel, revenue summed
    over every session with a purchase, and over each user's first purchase."""
    n = data.config.days
    cd = data.channel_daily
    spend = np.zeros((len(data.config.channels), n))
    spend[cd["channel"], cd["day"]] = cd["spend_cents"] / 100
    s = data.sessions
    revenue = np.bincount(s["day"], weights=s["revenue_cents"], minlength=n) / 100
    bought = np.flatnonzero(s["purchase"] >= 0)  # in time order
    _, first = np.unique(s["user_number"][bought], return_index=True)
    first = bought[first]
    new = np.bincount(s["day"][first], weights=s["revenue_cents"][first], minlength=n) / 100
    names = tuple(c.name for c in data.config.channels)
    return Inputs(data.dates, names, spend, revenue, new)


def _reader(path: Path, f, columns: tuple[str, ...]):
    """A CSV reader over ``f``, and the positions of ``columns`` in its header."""
    reader = csv.reader(f)
    header = next(reader, None)
    if header is None:
        raise ValueError(f"{path} is empty")
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"{path} has no {', '.join(missing)} column")
    return reader, len(header), [header.index(c) for c in columns]


def read_inputs(data_dir: Path) -> Inputs:
    """The model's inputs from the generator's CSVs: ``channel_daily.csv`` (spend only)
    and ``ga4_sessions.csv`` (purchase revenue by session date; a user id's earliest
    purchase is its first)."""
    data_dir = Path(data_dir)
    path = data_dir / "channel_daily.csv"
    spend_by: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as f:
        reader, width, (at, name, money) = _reader(path, f, ("date", "channel", "spend"))
        for row in reader:
            if len(row) != width:
                raise ValueError(
                    f"{path} line {reader.line_num} has {len(row)} fields, not {width}"
                )
            if row[at] in spend_by[row[name]]:
                raise ValueError(f"{path} has two rows for {row[name]} on {row[at]}")
            spend_by[row[name]][row[at]] = float(row[money])
    channels = tuple(spend_by)
    days = sorted({d for by_day in spend_by.values() for d in by_day})
    if not days:
        raise ValueError(f"{path} has no rows")
    start = dt.date.fromisoformat(days[0])
    n = (dt.date.fromisoformat(days[-1]) - start).days + 1
    dates = np.arange(np.datetime64(start, "D"), np.datetime64(start, "D") + n)
    index = {str(d): i for i, d in enumerate(dates)}
    spend = np.zeros((len(channels), n))
    for c, channel in enumerate(channels):
        if len(spend_by[channel]) != n:
            raise ValueError(f"{path} is missing a day for {channel}")
        for d, v in spend_by[channel].items():
            spend[c, index[d]] = v

    path = data_dir / "ga4_sessions.csv"
    revenue = np.zeros(n)
    first_purchase: dict[str, tuple[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        columns = ("session_start", "user_pseudo_id", "purchase_revenue")
        reader, width, (at, user, money) = _reader(path, f, columns)
        for row in reader:
            if len(row) != width:
                raise ValueError(
                    f"{path} line {reader.line_num} has {len(row)} fields, not {width}"
                )
            if row[money]:
                i = index.get(row[at][:10])
                if i is None:
                    raise ValueError(f"a session on {row[at][:10]} is outside the spend's days")
                value = float(row[money])
                revenue[i] += value
                seen = first_purchase.get(row[user])
                if seen is None or row[at] < seen[0]:
                    first_purchase[row[user]] = (row[at], value)
    new = np.zeros(n)
    for started, value in first_purchase.values():
        new[index[started[:10]]] += value
    return Inputs(dates, channels, spend, revenue, new)


def baseline_design(dates: np.ndarray, fourier_order: int = 2) -> tuple[np.ndarray, list[str]]:
    """Columns for everything but media: intercept, trend in years from the first day,
    yearly sine/cosine pairs, weekday indicators (Sunday the reference), and the Black
    Friday to Cyber Monday window."""
    days = dates.astype("datetime64[D]")
    t = (days - days[0]).astype(float)
    py = [dt.date.fromisoformat(str(d)) for d in days]
    doy = np.array([d.timetuple().tm_yday for d in py], dtype=float)
    cols, names = [np.ones(len(days)), t / 365.25], ["intercept", "trend"]
    for k in range(1, fourier_order + 1):
        angle = 2 * np.pi * k * doy / 365.25
        cols += [np.sin(angle), np.cos(angle)]
        names += [f"sin{k}", f"cos{k}"]
    weekday = np.array([d.weekday() for d in py])
    for i, name in enumerate(WEEKDAYS):
        cols.append((weekday == i).astype(float))
        names.append(name)
    holiday = np.array(
        [black_friday(d.year) <= d <= black_friday(d.year) + dt.timedelta(3) for d in py]
    )
    cols.append(holiday.astype(float))
    names.append("holiday")
    return np.column_stack(cols), names


@dataclass(frozen=True)
class ChannelFit:
    name: str
    decay: float
    half_saturation: float
    """Dollars of adstocked daily spend."""
    shape: float
    beta: float
    """Revenue a day at the curve's ceiling."""

    @property
    def curve(self) -> ResponseCurve:
        return ResponseCurve("hill", self.beta, self.half_saturation, self.shape)


@dataclass
class Fit:
    channels: list[ChannelFit]
    baseline_names: list[str]
    baseline_coef: np.ndarray
    train_days: int
    holdout_days: int
    fourier_order: int
    target: Target
    sse: float
    """Squared error on the training days."""
    evaluations: int
    seconds: float
    search: dict = field(default_factory=dict)

    def decompose(self, inputs: Inputs) -> dict[str, np.ndarray]:
        """Predicted revenue a day split into the baseline and each channel's share."""
        x, _ = baseline_design(inputs.dates, self.fourier_order)
        out = {"baseline": x @ self.baseline_coef}
        for c, ch in enumerate(self.channels):
            a = geometric_adstock(inputs.spend[c], ch.decay)
            out[ch.name] = ch.beta * hill(a, ch.half_saturation, ch.shape)
        return out

    def predict(self, inputs: Inputs) -> np.ndarray:
        return sum(self.decompose(inputs).values())


def _media(spend: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Columns hill(adstock(spend)), one per channel; params rows (decay, k, shape)."""
    return np.column_stack(
        [hill(geometric_adstock(s, d), k, sh) for s, (d, k, sh) in zip(spend, params, strict=True)]
    )


def _solve(
    base: np.ndarray,
    media: np.ndarray,
    y: np.ndarray,
    extra: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Least squares with the media coefficients held at 0 or above: the best of the
    unconstrained solutions over subsets of media columns whose coefficients come out
    non-negative (exact for a handful of channels).

    ``extra`` is further rows on the media coefficients alone, (rows, targets), solved
    with the rest; their squared residuals count in the returned error.
    """
    m = media.shape[1]
    if extra is not None:
        rows, targets = extra
        base = np.vstack([base, np.zeros((len(rows), base.shape[1]))])
        media = np.vstack([media, rows])
        y = np.concatenate([y, targets])
    best: tuple[float, np.ndarray] | None = None
    # Subsets in order of size, largest first; the full set usually wins outright.
    for mask in sorted(range(2**m), key=lambda b: -bin(b).count("1")):
        keep = [j for j in range(m) if mask >> j & 1]
        x = np.hstack([base, media[:, keep]])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        if (coef[base.shape[1] :] < 0).any():
            continue
        sse = float(((y - x @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            full = np.zeros(base.shape[1] + m)
            full[: base.shape[1]] = coef[: base.shape[1]]
            full[base.shape[1] + np.array(keep, dtype=int)] = coef[base.shape[1] :]
            best = (sse, full)
        if mask == 2**m - 1:
            break  # every coefficient non-negative: no subset can fit better
    assert best is not None  # the empty subset is always feasible
    return best[1], best[0]


def _to_unit(params: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Bounded parameters to unbounded search coordinates (logit of the position
    between the bounds; half-saturation and shape on a log scale)."""
    lo, hi = _bounds(scale)
    p = np.column_stack([params[:, 0], np.log(params[:, 1]), np.log(params[:, 2])])
    lo_, hi_ = (
        np.column_stack([lo[:, 0], np.log(lo[:, 1:])]),
        np.column_stack([hi[:, 0], np.log(hi[:, 1:])]),
    )
    u = np.clip((p - lo_) / (hi_ - lo_), 1e-6, 1 - 1e-6)
    return np.log(u / (1 - u)).ravel()


def _from_unit(z: np.ndarray, scale: np.ndarray) -> np.ndarray:
    lo, hi = _bounds(scale)
    u = 0.5 * (1 + np.tanh(z.reshape(-1, 3) / 2))  # the logistic, without overflow
    decay = lo[:, 0] + u[:, 0] * (hi[:, 0] - lo[:, 0])
    k = np.exp(np.log(lo[:, 1]) + u[:, 1] * (np.log(hi[:, 1]) - np.log(lo[:, 1])))
    s = np.exp(np.log(lo[:, 2]) + u[:, 2] * (np.log(hi[:, 2]) - np.log(lo[:, 2])))
    return np.column_stack([decay, k, s])


def _bounds(scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = len(scale)
    lo = np.column_stack(
        [
            np.full(m, DECAY_BOUNDS[0]),
            HALF_SATURATION_BOUNDS[0] * scale,
            np.full(m, SHAPE_BOUNDS[0]),
        ]
    )
    hi = np.column_stack(
        [
            np.full(m, DECAY_BOUNDS[1]),
            HALF_SATURATION_BOUNDS[1] * scale,
            np.full(m, SHAPE_BOUNDS[1]),
        ]
    )
    return lo, hi


def nelder_mead(
    f,
    x0: np.ndarray,
    *,
    step: float = 0.5,
    max_evals: int = 4000,
    tol: float = 1e-9,
    xtol: float = 1e-6,
):
    """Minimise ``f`` from ``x0`` with the Nelder-Mead simplex (standard coefficients).

    Stops after ``max_evals`` evaluations, or when the simplex's values agree to a
    relative ``tol`` *and* its points lie within ``xtol`` of the best in every
    coordinate: equal values alone can be points either side of the minimum. Returns
    (best point, its value, evaluations).
    """
    n = len(x0)
    pts = [np.asarray(x0, dtype=float)] + [x0 + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    evals = n + 1
    while evals < max_evals:
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        flat = abs(vals[-1] - vals[0]) <= tol * max(abs(vals[0]), 1e-300)
        if flat and max(np.abs(q - pts[0]).max() for q in pts[1:]) <= xtol:
            break
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = f(xr)
        evals += 1
        if fr < vals[0]:
            xe = centroid + 2 * (centroid - pts[-1])
            fe = f(xe)
            evals += 1
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            outside = fr < vals[-1]
            xc = centroid + 0.5 * ((xr if outside else pts[-1]) - centroid)
            fc = f(xc)
            evals += 1
            if fc < (fr if outside else vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
                evals += n
    i = int(np.argmin(vals))
    return pts[i], vals[i], evals


@dataclass(frozen=True)
class Prior:
    """Log-normal priors on every searched channel's shape and half-saturation.

    The search then minimises the training error plus ``noise_variance`` times the sum,
    over channels, of ``(log(shape / self.shape) / sd) ** 2`` and of
    ``(log(half_saturation / (self.half_saturation * mean spend)) / sd) ** 2``: for
    Gaussian errors, the noise variance times twice the negative log posterior (on the
    log scale of both parameters), up to a constant. The noise variance is
    the residual variance of the baseline terms alone on the training rows, fixed before
    the search, so the prior's weight does not depend on how well the curves fit. The
    default centre is a curve without an S (shape 1) half-saturated at the channel's
    current budget.
    """

    shape: float = 1.0
    half_saturation: float = 1.0
    """As a multiple of the channel's mean daily spend over the training days."""
    sd: float = 0.5
    """Of the logarithm: 0.5 puts two thirds of the prior within a factor of 1.65."""

    def __post_init__(self) -> None:
        for name in ("shape", "half_saturation", "sd"):
            value = getattr(self, name)
            if (
                isinstance(value, bool | np.bool_)
                or not isinstance(value, numbers.Real)
                or not 0 < value < math.inf
            ):
                raise ValueError(f"prior {name} must be a positive finite number, got {value!r}")


@dataclass(frozen=True)
class Lift:
    """A measured lift for one channel, as from a holdout experiment: the revenue the
    channel brought over the days ``start`` to ``end`` (inclusive), with its standard
    error. The fit is then asked to match it, through one more least-squares row on the
    channel's coefficient weighted by ``sqrt(noise variance) / se``, so a miss adds the
    noise variance times its square in standard errors to the training error (the noise
    variance as for ``Prior``)."""

    channel: str
    start: np.datetime64
    end: np.datetime64
    revenue: float
    se: float

    def __post_init__(self) -> None:
        for name in ("revenue", "se"):
            value = getattr(self, name)
            if (
                isinstance(value, bool | np.bool_)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
                or value < 0
                or (name == "se" and value == 0)
            ):
                rule = "above 0" if name == "se" else "at least 0"
                raise ValueError(f"lift {name} must be a finite number, {rule}, got {value!r}")
        if np.datetime64(self.end, "D") < np.datetime64(self.start, "D"):
            raise ValueError("a lift's end comes before its start")


def _weeks(n: int) -> np.ndarray:
    """A (weeks, n) matrix summing each full week of n days from the first; a last part
    week is left out."""
    w = n // 7
    out = np.zeros((w, n))
    for i in range(w):
        out[i, 7 * i : 7 * i + 7] = 1.0
    return out


def fit(
    inputs: Inputs,
    *,
    holdout_days: int = 91,
    target: Target = "new_customer",
    fourier_order: int = 2,
    fixed: dict[str, tuple[float, float, float]] | None = None,
    max_evals: int = 4000,
    prior: Prior | None = None,
    lifts: tuple[Lift, ...] = (),
    weekly: bool = False,
) -> Fit:
    """Fit the model on all but the last ``holdout_days`` days.

    ``fixed`` gives some channels' (decay, half-saturation in dollars, shape) instead of
    searching for them, for instance the true ones, to see how well the regression does
    when the curves are right. ``prior`` adds priors on the searched curves (see
    ``Prior``), ``lifts`` measured lifts the fit must also match (see ``Lift``; they must
    fall in the training days), and ``weekly`` fits weekly sums of revenue and of each
    channel's daily response instead of days (the weekday terms then drop out, and so do
    the training days after the last full week).
    """
    if (
        not isinstance(holdout_days, numbers.Integral)
        or isinstance(holdout_days, bool | np.bool_)
        or holdout_days < 0
    ):
        raise ValueError(f"holdout_days must be a whole number, at least 0, got {holdout_days!r}")
    holdout_days = int(holdout_days)
    n_train = len(inputs) - holdout_days
    base_all, names = baseline_design(inputs.dates, fourier_order)
    if n_train < base_all.shape[1] + 4 * len(inputs.channels):
        raise ValueError(f"{n_train} training days are too few for the model's parameters")
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(inputs.channels)
    if unknown:
        raise ValueError(f"fixed names channels the inputs do not have: {sorted(unknown)}")
    if prior is not None and not isinstance(prior, Prior):
        raise TypeError(f"prior must be a Prior, not {type(prior).__name__}")
    started = time.perf_counter()
    base, y = base_all[:n_train], inputs.target(target)[:n_train]
    spend = inputs.spend
    scale = spend[:, :n_train].mean(axis=1)
    if (scale <= 0).any():
        raise ValueError("every channel needs some spend on the training days")
    first = inputs.dates[0].astype("datetime64[D]")
    windows = []
    for lift in lifts:
        if not isinstance(lift, Lift):
            raise TypeError(f"lifts must be Lift, not {type(lift).__name__}")
        if lift.channel not in inputs.channels:
            raise ValueError(f"a lift names a channel the inputs do not have: {lift.channel!r}")
        i0 = int((np.datetime64(lift.start, "D") - first).astype(int))
        i1 = int((np.datetime64(lift.end, "D") - first).astype(int)) + 1
        if i0 < 0 or i1 > n_train:
            raise ValueError(
                f"a lift for {lift.channel} from {lift.start} to {lift.end} is not within "
                "the training days"
            )
        windows.append((inputs.channels.index(lift.channel), i0, i1, lift))
    if weekly:
        summed = _weeks(n_train)
        if len(summed) < base_all.shape[1] - len(WEEKDAYS) + 4 * len(inputs.channels):
            raise ValueError(f"{len(summed)} training weeks are too few for the model's parameters")
        kept = [j for j, name in enumerate(names) if name not in WEEKDAYS]
        base_fit, y_fit = summed @ base[:, kept], summed @ y
    else:
        summed, kept = None, list(range(len(names)))
        base_fit, y_fit = base, y
    if prior is not None or windows:
        c0, *_ = np.linalg.lstsq(base_fit, y_fit, rcond=None)
        resid = y_fit - base_fit @ c0
        noise = float(resid @ resid) / (len(y_fit) - np.linalg.matrix_rank(base_fit))
    else:
        noise = None
    evals = 0
    cache: dict[tuple, np.ndarray] = {}
    caching = True

    def column(c: int, p) -> np.ndarray:
        # Cached during the grid, where most columns repeat; the simplex rarely revisits
        # a point, so caching there would only hold memory.
        key = (c, *p)
        if key in cache:
            return cache[key]
        col = hill(geometric_adstock(spend[c], p[0]), p[1], p[2])[:n_train]
        if caching:
            cache[key] = col
        return col

    def extra(media: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """One row per lift: the channel's summed response over its window."""
        if not windows:
            return None
        rows = np.zeros((len(windows), media.shape[1]))
        targets = np.zeros(len(windows))
        for r, (c, i0, i1, lift) in enumerate(windows):
            weight = math.sqrt(noise) / lift.se
            rows[r, c] = media[i0:i1, c].sum() * weight
            targets[r] = lift.revenue * weight
        return rows, targets

    def penalty(params: np.ndarray) -> float:
        if prior is None:
            return 0.0
        total = 0.0
        for c in free:
            _, k, s = params[c]
            total += math.log(s / prior.shape) ** 2
            total += math.log(k / (prior.half_saturation * scale[c])) ** 2
        return noise * total / prior.sd**2

    def objective(params: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        media = np.column_stack([column(c, tuple(p)) for c, p in enumerate(params)])
        rows = media if summed is None else summed @ media
        return _solve(base_fit, rows, y_fit, extra(media))[1] + penalty(params)

    free = [c for c, name in enumerate(inputs.channels) if name not in fixed]
    params = np.array(
        [
            fixed[name] if name in fixed else (0.4, 1.0 * scale[c], 1.0)
            for c, name in enumerate(inputs.channels)
        ],
        dtype=float,
    )
    # Coarse grid, one channel at a time, twice round.
    for _ in range(2):
        for c in free:
            best = (math.inf, params[c].copy())
            for d in GRID_DECAY:
                for k in GRID_HALF_SATURATION:
                    for s in GRID_SHAPE:
                        params[c] = (d, k * scale[c], s)
                        v = objective(params)
                        if v < best[0]:
                            best = (v, params[c].copy())
            params[c] = best[1]
    grid_value = objective(params)
    grid_params = params.copy()
    cache.clear()
    caching = False
    if free:
        sub = scale[free]

        def f(z: np.ndarray) -> float:
            p = params.copy()
            p[free] = _from_unit(z, sub)
            return objective(p)

        z, _, _ = nelder_mead(f, _to_unit(params[free], sub), max_evals=max_evals)
        params[free] = _from_unit(z, sub)
    media = _media(spend[:, :n_train], params)
    rows = media if summed is None else summed @ media
    coef, final = _solve(base_fit, rows, y_fit, extra(media))
    nb = base_fit.shape[1]
    baseline_coef = np.zeros(len(names))
    baseline_coef[kept] = coef[:nb]
    betas = coef[nb:]
    if summed is None and prior is None and not windows:
        sse = final
    else:
        sse = float(((y - base @ baseline_coef - media @ betas) ** 2).sum())
    channels = [
        ChannelFit(name, float(p[0]), float(p[1]), float(p[2]), float(betas[c]))
        for c, (name, p) in enumerate(zip(inputs.channels, params, strict=True))
    ]
    return Fit(
        channels=channels,
        baseline_names=names,
        baseline_coef=baseline_coef,
        train_days=n_train,
        holdout_days=holdout_days,
        fourier_order=fourier_order,
        target=target,
        sse=sse,
        evaluations=evals,
        seconds=time.perf_counter() - started,
        search={
            "grid_sse": grid_value,
            "grid_params": grid_params.tolist(),
            "objective": final + penalty(params),
            "noise_variance": noise,
        },
    )


def fit_baseline_only(
    inputs: Inputs,
    *,
    holdout_days: int = 91,
    target: Target = "new_customer",
    fourier_order: int = 2,
) -> np.ndarray:
    """Predicted revenue from the baseline terms alone, fitted on the same training days:
    what the media terms have to beat."""
    x, _ = baseline_design(inputs.dates, fourier_order)
    n_train = len(inputs) - holdout_days
    coef, *_ = np.linalg.lstsq(x[:n_train], inputs.target(target)[:n_train], rcond=None)
    return x @ coef


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    """R² (1 - SSE / total sum of squares about the actual mean), MAPE and RMSE.

    MAPE is over the days with revenue (``mape_days`` of them), since a day without any
    has no percentage error; R² is None when the actual values do not vary (one day, say),
    and MAPE when no day has revenue. None, not NaN, so the figures stay valid JSON.
    """
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if len(actual) == 0:
        raise ValueError("no days to score")
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must be the same length")
    resid = actual - predicted
    total = ((actual - actual.mean()) ** 2).sum()
    sold = actual > 0
    return {
        "r2": float(1 - (resid**2).sum() / total) if total > 0 else None,
        "mape": float(np.mean(np.abs(resid[sold]) / actual[sold])) if sold.any() else None,
        "mape_days": int(sold.sum()),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "days": len(actual),
    }


def evaluate(model: Fit, inputs: Inputs, predicted: np.ndarray | None = None) -> dict[str, dict]:
    """Scores on the training days and, separately, on the held-out days."""
    p = model.predict(inputs) if predicted is None else predicted
    n = model.train_days
    y = inputs.target(model.target)
    out = {"train": scores(y[:n], p[:n])}
    if model.holdout_days:
        out["holdout"] = scores(y[n:], p[n:])
    return out


# --- comparison with the generator's truth (never used by the fit) --------------------------


def read_truth(data_dir: Path, dates: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    """truth.json, and each component's caused revenue on each of ``dates`` from
    truth_daily.csv, which must have exactly one row per component and day."""
    data_dir = Path(data_dir)
    truth = json.loads((data_dir / "truth.json").read_text(encoding="utf-8"))
    path = data_dir / "truth_daily.csv"
    index = {str(d): i for i, d in enumerate(dates.astype("datetime64[D]"))}
    revenue: dict[str, np.ndarray] = {}
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as f:
        reader, width, (at, name, money) = _reader(path, f, ("date", "component", "revenue"))
        for row in reader:
            if len(row) != width:
                raise ValueError(
                    f"{path} line {reader.line_num} has {len(row)} fields, not {width}"
                )
            if row[at] not in index:
                raise ValueError(f"{path} has a row on {row[at]}, outside the inputs' days")
            if row[at] in seen[row[name]]:
                raise ValueError(f"{path} has two rows for {row[name]} on {row[at]}")
            seen[row[name]].add(row[at])
            revenue.setdefault(row[name], np.zeros(len(index)))[index[row[at]]] = float(row[money])
    for name, days in seen.items():
        if len(days) != len(index):
            raise ValueError(f"{path} has {len(days)} of the inputs' {len(index)} days for {name}")
    return truth, revenue


def compare_with_truth(
    model: Fit, inputs: Inputs, truth: dict, caused: dict[str, np.ndarray]
) -> list[dict]:
    """Per channel: fitted against true decay, half-saturation and shape, and the
    revenue the fit credits to the channel against the revenue it caused, on the
    training days and on the held-out days."""
    parts = model.decompose(inputs)
    n = model.train_days
    rows = []
    for ch in model.channels:
        t = truth["channels"][ch.name]
        rows.append(
            {
                "channel": ch.name,
                "decay": (ch.decay, t["decay"]),
                "half_saturation": (ch.half_saturation, t["half_saturation_dollars"]),
                "shape": (ch.shape, t["hill_shape"]),
                "revenue_train": (
                    float(parts[ch.name][:n].sum()),
                    float(caused[ch.name][:n].sum()),
                ),
                "revenue_holdout": (
                    float(parts[ch.name][n:].sum()),
                    float(caused[ch.name][n:].sum()),
                ),
            }
        )
    return rows


def expected_new_revenue(data_dir: Path, dates: np.ndarray, truth: dict) -> np.ndarray:
    """The new-customer revenue each day was expected to bring: the purchases the
    generator expected from each channel and the baseline (all first purchases), times
    the mean order value. Scoring it against the actual revenue shows how much of the
    day-to-day variation is chance that no model of spend can explain."""
    path = Path(data_dir) / "truth_daily.csv"
    index = {str(d): i for i, d in enumerate(dates.astype("datetime64[D]"))}
    purchases = np.zeros(len(index))
    with path.open(newline="", encoding="utf-8") as f:
        reader, width, (at, expected) = _reader(path, f, ("date", "expected_conversions"))
        for row in reader:
            if len(row) != width:
                raise ValueError(
                    f"{path} line {reader.line_num} has {len(row)} fields, not {width}"
                )
            if row[expected]:  # repeat purchases have no expected value
                purchases[index[row[at]]] += float(row[expected])
    return purchases * float(truth["config"]["aov"])


def true_params(truth: dict) -> dict[str, tuple[float, float, float]]:
    return {
        name: (c["decay"], c["half_saturation_dollars"], c["hill_shape"])
        for name, c in truth["channels"].items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m marketing_roi.mmm",
        description="Fit the mix model to the generated data and score it on held-out days.",
    )
    parser.add_argument("--data", type=Path, default=Path("data/generated"))
    parser.add_argument("--holdout-days", type=int, default=91)
    parser.add_argument(
        "--target",
        choices=("new_customer", "total"),
        default="new_customer",
        help="revenue to model: first purchases only (default) or all purchases",
    )
    parser.add_argument("--json", type=Path, help="also write the figures to this file")
    args = parser.parse_args(argv)
    try:
        inputs = read_inputs(args.data)
        truth, caused = read_truth(args.data, inputs.dates)
        known = true_params(truth)
        expected = expected_new_revenue(args.data, inputs.dates, truth)
        missing = [c for c in inputs.channels if c not in known or c not in caused]
        if missing:
            raise ValueError(f"the truth has no record of {', '.join(missing)}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: cannot read {args.data}: {exc}", file=sys.stderr)
        return 1
    options = {"holdout_days": args.holdout_days, "target": args.target}
    try:
        model = fit(inputs, **options)
    except ValueError as exc:
        parser.error(str(exc))
    oracle = fit(inputs, **options, fixed={c: known[c] for c in inputs.channels})
    figures = {
        "period": [str(inputs.dates[0]), str(inputs.dates[-1])],
        "target": args.target,
        "train_days": model.train_days,
        "holdout_days": model.holdout_days,
        "fit": evaluate(model, inputs),
        "true_curves": evaluate(oracle, inputs),
        "baseline_only": evaluate(model, inputs, fit_baseline_only(inputs, **options)),
        "expected_revenue": evaluate(model, inputs, expected)
        if args.target == "new_customer"
        else None,
        "channels": compare_with_truth(model, inputs, truth, caused),
        "true_curves_channels": compare_with_truth(oracle, inputs, truth, caused),
        "search": {
            "evaluations": model.evaluations,
            "seconds": round(model.seconds, 1),
            "grid_sse": model.search["grid_sse"],
            "sse": model.sse,
            "true_curves_sse": oracle.sse,
        },
    }
    print(
        f"{figures['period'][0]} to {figures['period'][1]}, {args.target} revenue: "
        f"{model.train_days} training days, {model.holdout_days} held out"
    )
    for label, key in (
        ("fitted model", "fit"),
        ("same regression, true curves", "true_curves"),
        ("baseline terms only", "baseline_only"),
        ("truth's expected revenue", "expected_revenue"),
    ):
        if figures[key] is None:
            continue
        for part, sc in figures[key].items():
            r2 = "n/a" if sc["r2"] is None else f"{sc['r2']:.3f}"
            mape = "n/a" if sc["mape"] is None else f"{sc['mape']:.1%}"
            print(
                f"  {label:28s} {part:8s} R2 {r2}  MAPE {mape}"
                f"  RMSE ${sc['rmse']:,.0f}  ({sc['days']} days)"
            )
    print(
        "  fitted (true)   decay        half-sat $          shape        revenue credited / caused"
    )
    for r in figures["channels"]:
        line = (
            f"  {r['channel']:13s} {r['decay'][0]:.2f} ({r['decay'][1]:.2f})"
            f"  {r['half_saturation'][0]:>7,.0f} ({r['half_saturation'][1]:,.0f})"
            f"  {r['shape'][0]:.2f} ({r['shape'][1]:.2f})"
            f"  train ${r['revenue_train'][0]:,.0f} / ${r['revenue_train'][1]:,.0f}"
        )
        if model.holdout_days:
            line += f", holdout ${r['revenue_holdout'][0]:,.0f} / ${r['revenue_holdout'][1]:,.0f}"
        print(line)
    print(f"  search: {model.evaluations:,} fits in {model.seconds:.1f} s")
    if args.json:
        text = json.dumps(figures, indent=2, allow_nan=False)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
