"""Holdout validation of a reallocation: what moving the budget would have done over the
held-out days, as each fitted model projects it and as the generator's truth has it.

The optimiser splits the training days' mean daily budget, spent steadily. A business
would apply the split to the next quarter's budget, which keeps its own flights,
quarterly level and daily pattern. So to try a split on the held-out days, each
channel's daily spend there is multiplied by its split's budget over its current one,
and then every channel by one common factor, so that the held-out days' total spend is
what it was: only the shares between channels change. Spend before the held-out days is
untouched, so the carryover into them is too.

The **projected** lift is what a fit expects from that: its predicted revenue over the
held-out days with the reallocated spend, less with the spend as it was (the baseline
cancels). The **true** lift is the same difference on the generator's expected revenue,
which no real analysis could see. Both are revenue over the held-out days. As an ROI
lift they are over the held-out days' spend, which the reallocation keeps the same: the
change in (revenue - spend) / spend is the extra revenue over the spend.

The held-out days are also the only days the fit never saw, so how far off its forecast
of their revenue was, with the spend as it was, is reported beside: a before-and-after
reading of a real reallocation would have to tell its lift apart from an error that size.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

from marketing_roi import mmm
from marketing_roi.generate import adstock
from marketing_roi.saturation import ResponseCurve


def reallocated(
    spend: np.ndarray,
    start: int,
    channels: Sequence[str],
    current: Mapping[str, float],
    split: Mapping[str, float],
) -> np.ndarray:
    """``spend`` (channels, days) with each channel's days from ``start`` on scaled by
    ``split[c] / current[c]``, then all of them by one factor so their total is unchanged.
    """
    spend = np.asarray(spend, dtype=float)
    if not 0 <= start < spend.shape[1]:
        raise ValueError(f"start must be a day of the spend, got {start}")
    if set(current) != set(channels) or set(split) != set(channels):
        raise ValueError("current and split must name the spend's channels")
    if any(current[c] <= 0 for c in channels):
        raise ValueError("every channel's current budget must be above 0")
    factors = np.array([split[c] / current[c] for c in channels])
    after = spend[:, start:]
    scaled = after * factors[:, None]
    if scaled.sum() <= 0:
        raise ValueError("the split leaves no spend on the held-out days")
    out = spend.copy()
    out[:, start:] = scaled * (after.sum() / scaled.sum())
    return out


def fitted_revenue(model: mmm.Fit, inputs: mmm.Inputs, spend: np.ndarray) -> np.ndarray:
    """The fit's predicted revenue a day with ``spend`` in place of the inputs' own."""
    return model.predict(dataclasses.replace(inputs, spend=spend))


def true_media_revenue(
    curves: Mapping[str, ResponseCurve], decays: Mapping[str, float], spend: np.ndarray
) -> np.ndarray:
    """The generator's expected revenue a day from the channels at ``spend``: each true
    revenue curve over the generator's adstock (which starts as if the first day's spend
    had run for ever)."""
    return sum(
        np.asarray(curves[c].response(adstock(spend[i], decays[c])), dtype=float)
        for i, c in enumerate(curves)
    )


def validate(
    inputs: mmm.Inputs,
    models: Mapping[str, mmm.Fit],
    true_curves: Mapping[str, ResponseCurve],
    decays: Mapping[str, float],
    current: Mapping[str, float],
    splits: Mapping[str, Mapping[str, float]],
    holdout_days: int,
) -> dict:
    """Each split in ``splits`` applied to the last ``holdout_days`` days: the lift every
    fit in ``models`` projects and the true lift, in dollars over those days, as a share
    of their new-customer revenue, and as an ROI lift over their spend; and each fit's
    forecast of those days' new-customer revenue at the spend as it was."""
    channels = inputs.channels
    if list(true_curves) != list(channels):
        raise ValueError("the true curves must be in the inputs' channel order")
    start = len(inputs) - holdout_days
    if not 0 < start < len(inputs):
        raise ValueError(f"holdout_days must leave training days, got {holdout_days}")
    held = slice(start, None)
    spend = float(inputs.spend[:, held].sum())
    revenue = float(inputs.new_revenue[held].sum())
    as_spent = {name: fitted_revenue(m, inputs, inputs.spend) for name, m in models.items()}
    true_as_spent = true_media_revenue(true_curves, decays, inputs.spend)
    out: dict = {
        "days": holdout_days,
        "first_day": str(inputs.dates[start]),
        "spend": spend,
        "new_customer_revenue": revenue,
        "forecast": {
            name: {
                "predicted": float(p[held].sum()),
                "error_share": float(p[held].sum()) / revenue - 1,
            }
            for name, p in as_spent.items()
        },
        "splits": {},
    }
    for label, split in splits.items():
        new = reallocated(inputs.spend, start, channels, current, split)
        true = float((true_media_revenue(true_curves, decays, new) - true_as_spent)[held].sum())
        projected = {
            name: float((fitted_revenue(m, inputs, new) - as_spent[name])[held].sum())
            for name, m in models.items()
        }
        out["splits"][label] = {
            "spend": {c: float(new[i, held].sum()) for i, c in enumerate(channels)},
            "projected": projected,
            "projected_share": {name: v / revenue for name, v in projected.items()},
            "projected_roi": {name: v / spend for name, v in projected.items()},
            "true": true,
            "true_share": true / revenue,
            "true_roi": true / spend,
        }
    return out
