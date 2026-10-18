"""Budget reallocation: split a fixed daily budget over the channels' fitted response
curves so that the expected revenue is as high as it can be.

At a steady daily spend ``x`` a channel's adstock equals ``x`` (the transforms are
normalised), so its fitted curve ``beta * hill(x, k, s)`` is its expected revenue a day
at that budget. ``allocate`` maximises the sum of those over the channels, with the
total fixed and each channel kept between a floor and a cap. With a shape above 1 a
Hill curve is an S, the problem is not concave, and equalising marginal returns can
find a minimum as easily as a maximum; so ``allocate`` solves it exactly on a grid of
``steps`` equal parts of the total (by dynamic programming over the channels), and is
right to within one part, ``total / steps`` dollars a day per channel.

The floor and cap (by default half and twice each channel's current budget) keep the
answer where the curves were fitted: a curve says little about spend far outside what
the channel actually spent.

The fitted curves are not recovered (see ``identify``): fits that the held-out data
cannot tell apart credit the channels very differently. So ``python -m
marketing_roi.optimise`` does not report one reallocation. It fits the run every way
``identify`` does that uses no truth beyond a measured lift, optimises under each fit,
and reports the spread: each channel's recommended budget across the fits, how many
fits agree on the direction of the change, and the **projected** revenue gain each fit
expects from its own reallocation. Because the data are generated, it also scores each
reallocation on the true curves, which no real analysis could do: that is the gain the
reallocation would really bring, and the true curves' own optimum is the most any could.

Every gain is measured from the current budgets spent steadily, which is not how they
were spent (flights, quarterly levels, daily noise); the true revenue of both is
reported, so the part of a gain that comes from spending steadily at all can be seen.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from marketing_roi import identify, mmm
from marketing_roi.generate import GeneratorConfig, adstock, generate
from marketing_roi.saturation import ResponseCurve, true_curve

FITS = tuple(v for v in identify.VARIANTS if v != "true_curves")
"""Every way of fitting that uses no truth beyond the measured lift."""

TRUTHS: dict[str, tuple[float, float] | None] = {
    "default": None,
    "moved": (2.5, 2.5),
    "low": (0.7, 0.5),
}
"""The generator's channels as they are, or every channel's true (shape, half-saturation
as a multiple of daily spend) set away from the priors' centre of (1, 1): an S that
saturates late, or a curve that saturates early with no S."""


@dataclass(frozen=True)
class Allocation:
    spend: dict[str, float]
    """Dollars a day per channel; they sum to the total."""
    revenue: float
    """Expected revenue a day from the channels at this spend, on the curves used."""
    marginal: dict[str, float]
    """Extra revenue a day from one more dollar a day, per channel, at this spend."""
    resolution: float
    """The grid's step, dollars a day: the answer is exact to within it per channel."""


def _number(name: str, value) -> float:
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite number, at least 0, got {value!r}")
    return float(value)


def allocate(
    curves: Mapping[str, ResponseCurve],
    total: float,
    lower: Mapping[str, float] | None = None,
    upper: Mapping[str, float] | None = None,
    *,
    steps: int = 4000,
) -> Allocation:
    """The split of ``total`` dollars a day over ``curves`` with the most revenue, each
    channel between its ``lower`` and ``upper`` budget (0 and ``total`` if not given).

    Exact on a grid of ``steps`` parts of the total; a bound off the grid is moved
    inwards to the nearest grid point.
    """
    names = list(curves)
    if not names:
        raise ValueError("no channels to allocate to")
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 1:
        raise ValueError(f"steps must be a whole number, at least 1, got {steps!r}")
    total = _number("total", total)
    if total == 0:
        raise ValueError("total must be above 0")
    lower, upper = dict(lower or {}), dict(upper or {})
    unknown = (set(lower) | set(upper)) - set(names)
    if unknown:
        raise ValueError(f"bounds name channels without a curve: {sorted(unknown)}")
    unit = total / steps
    lo, hi = [], []
    for name in names:
        a = _number(f"lower bound for {name}", lower.get(name, 0.0))
        b = _number(f"upper bound for {name}", upper.get(name, total))
        if b < a:
            raise ValueError(f"{name}'s upper bound is below its lower bound")
        lo.append(math.ceil(a / unit - 1e-9))
        hi.append(min(math.floor(b / unit + 1e-9), steps))
        if hi[-1] < lo[-1]:
            raise ValueError(f"{name}'s bounds hold no point of the grid")
    if sum(lo) > steps or sum(hi) < steps:
        raise ValueError("the bounds cannot add up to the total")

    grid = np.arange(steps + 1) * unit
    # best[b]: most revenue from the channels so far spending b units; picks[c][b]: units
    # channel c spends in that best split.
    best = np.full(steps + 1, -math.inf)
    first = np.asarray(curves[names[0]].response(grid[lo[0] : hi[0] + 1]), dtype=float)
    best[lo[0] : hi[0] + 1] = first
    picks = [np.arange(steps + 1)]
    for c in range(1, len(names)):
        f = np.asarray(curves[names[c]].response(grid[lo[c] : hi[c] + 1]), dtype=float)
        new = np.full(steps + 1, -math.inf)
        pick = np.zeros(steps + 1, dtype=int)
        for i, j in enumerate(range(lo[c], hi[c] + 1)):
            candidate = best[: steps + 1 - j] + f[i]
            better = candidate > new[j:]
            new[j:][better] = candidate[better]
            pick[j:][better] = j
        best, picks = new, [*picks, pick]
    units, b = [0] * len(names), steps
    for c in range(len(names) - 1, -1, -1):
        units[c] = int(picks[c][b])
        b -= units[c]
    assert b == 0
    spend = {name: u * unit for name, u in zip(names, units, strict=True)}
    return Allocation(
        spend=spend,
        revenue=float(sum(curves[n].response(spend[n]) for n in names)),
        marginal={n: float(curves[n].marginal(spend[n])) for n in names},
        resolution=unit,
    )


def steady_revenue(curves: Mapping[str, ResponseCurve], spend: Mapping[str, float]) -> float:
    """Expected revenue a day from the channels at a steady ``spend``."""
    return float(sum(curves[n].response(spend[n]) for n in curves))


def plan(
    seed: int = 42,
    *,
    fits: tuple[str, ...] = FITS,
    floor: float = 0.5,
    cap: float = 2.0,
    holdout_days: int = 91,
    steps: int = 4000,
    days: int = 731,
    truth: str = "default",
) -> dict:
    """Fit one generated run each way in ``fits``, reallocate its current budget under
    each, and score every reallocation on the true curves too."""
    if not 0 <= floor <= 1 <= cap:
        raise ValueError(f"need 0 <= floor <= 1 <= cap, got floor {floor}, cap {cap}")
    if not math.isfinite(cap):
        raise ValueError(f"cap must be finite, got {cap}")
    if floor == cap:
        raise ValueError("floor and cap are both 1: there is nothing to reallocate")
    if isinstance(holdout_days, bool) or not isinstance(holdout_days, numbers.Integral):
        raise ValueError(f"holdout_days must be a whole number, got {holdout_days!r}")
    if not 0 <= holdout_days < days:
        raise ValueError(
            f"holdout_days must be at least 0 and below the {days} days, got {holdout_days}"
        )
    unknown = [f for f in fits if f not in FITS]
    if unknown or not fits:
        raise ValueError(f"unknown fits {unknown}; choose from {', '.join(FITS)}")
    if truth not in TRUTHS:
        raise ValueError(f"unknown truth {truth!r}; choose from {', '.join(TRUTHS)}")
    config = GeneratorConfig(seed=seed, days=days)
    if TRUTHS[truth] is not None:
        config = identify.moved(config, *TRUTHS[truth])
    data = generate(config)
    inputs = mmm.inputs_from_dataset(data)
    caused = identify.caused_revenue(data)
    n = len(inputs) - holdout_days
    current = {c: float(inputs.spend[i, :n].mean()) for i, c in enumerate(inputs.channels)}
    total = sum(current.values())
    lower = {c: floor * x for c, x in current.items()}
    upper = {c: cap * x for c, x in current.items()}
    true_curves = {
        ch.name: _revenue_curve(true_curve(ch, data.betas[ch.name]), data.config.aov)
        for ch in data.config.channels
    }
    best_true = allocate(true_curves, total, lower, upper, steps=steps)
    true_now = steady_revenue(true_curves, current)
    # The true revenue of the budgets as they were really spent (flights, quarterly
    # levels, daily noise), against the same budgets spent steadily: the gains below
    # are all over the steady spend.
    true_actual = sum(
        float(true_curves[ch.name].response(adstock(inputs.spend[i], ch.decay))[:n].mean())
        for i, ch in enumerate(data.config.channels)
    )
    revenue = float(inputs.new_revenue[:n].mean())
    out: dict = {
        "seed": seed,
        "truth": truth,
        "train_days": n,
        "current": current,
        "total": total,
        "floor": floor,
        "cap": cap,
        "resolution": total / steps,
        "new_customer_revenue": revenue,
        "observed_spend": {
            c: [float(inputs.spend[i, :n].min()), float(inputs.spend[i, :n].max())]
            for i, c in enumerate(inputs.channels)
        },
        "true_optimum": {
            "spend": best_true.spend,
            "true_gain": best_true.revenue - true_now,
            "true_gain_share": (best_true.revenue - true_now) / revenue,
            "true_now": true_now,
            "true_actual": true_actual,
            "steady_gain": true_now - true_actual,
            "steady_gain_share": (true_now - true_actual) / revenue,
        },
        "fits": {},
    }
    for name in fits:
        model = mmm.fit(inputs, **identify.options(name, data, caused, holdout_days))
        curves = {ch.name: ch.curve for ch in model.channels}
        best = allocate(curves, total, lower, upper, steps=steps)
        now = steady_revenue(curves, current)
        out["fits"][name] = {
            "spend": best.spend,
            "projected_gain": best.revenue - now,
            "projected_gain_share": (best.revenue - now) / revenue,
            "media_now": now,
            # None where the slope is infinite (a shape below 1 at no spend), for valid JSON.
            "marginal": {c: m if math.isfinite(m) else None for c, m in best.marginal.items()},
            "true_gain": steady_revenue(true_curves, best.spend) - true_now,
            "true_gain_share": (steady_revenue(true_curves, best.spend) - true_now) / revenue,
            "holdout_mape": mmm.evaluate(model, inputs)["holdout"]["mape"]
            if holdout_days
            else None,
            "seconds": round(model.seconds, 1),
        }
    out["spread"] = spread(out)
    return out


def _revenue_curve(purchases: ResponseCurve, aov: float) -> ResponseCurve:
    """A purchase curve as revenue: beta times the mean order value."""
    return ResponseCurve("hill", purchases.beta * aov, purchases.half_saturation, purchases.shape)


def spread(result: dict) -> dict:
    """Per channel, the recommended budget's range over the fits and how many fits
    move it up or down (by more than the grid's resolution); and the range of the
    projected and the true gains."""
    fits = result["fits"].values()
    tol = result["resolution"]
    out: dict = {"channels": {}}
    for c, now in result["current"].items():
        values = [f["spend"][c] for f in fits]
        out["channels"][c] = {
            "min": min(values),
            "median": float(np.median(values)),
            "max": max(values),
            "up": sum(v > now + tol for v in values),
            "down": sum(v < now - tol for v in values),
            "true_optimum": result["true_optimum"]["spend"][c],
        }
    for key in ("projected_gain", "true_gain"):
        values = [f[key] for f in fits]
        out[key] = {"min": min(values), "median": float(np.median(values)), "max": max(values)}
    out["fits"] = len(result["fits"])
    out["true_gain_below_zero"] = sum(f["true_gain"] < 0 for f in fits)
    return out


def summarise(results: list[dict]) -> dict:
    """Per fit, over the runs: the median projected and true gain as shares of the
    run's new-customer revenue, how often the true gain fell below zero, and the median
    share of the most any reallocation could gain that the fit's reallocation got."""
    out = {}
    for name in results[0]["fits"]:
        fits = [r["fits"][name] for r in results]
        captured = [
            f["true_gain"] / r["true_optimum"]["true_gain"]
            for f, r in zip(fits, results, strict=True)
            if r["true_optimum"]["true_gain"] > 0
        ]
        out[name] = {
            "projected_share": float(np.median([f["projected_gain_share"] for f in fits])),
            "true_share": float(np.median([f["true_gain_share"] for f in fits])),
            "true_below_zero": sum(f["true_gain"] < 0 for f in fits),
            "captured": float(np.median(captured)) if captured else None,
            "runs": len(fits),
        }
    shares = [r["true_optimum"]["true_gain_share"] for r in results]
    steady = [r["true_optimum"]["steady_gain_share"] for r in results]
    out["true_optimum"] = {
        "true_share": float(np.median(shares)),
        "steady_share": float(np.median(steady)),
        "runs": len(results),
    }
    return out


def _truth_label(truth: str) -> str:
    if truth == "default":
        return ""
    shape, half = TRUTHS[truth]
    return f" ({truth} truth: shape {shape:g}, half-saturation {half:g}x)"


def _print_run(result: dict) -> None:
    current, s = result["current"], result["spread"]
    print(
        f"seed {result['seed']}{_truth_label(result['truth'])}: ${result['total']:,.0f} a day "
        f"over {len(current)} channels, "
        f"each kept within {result['floor']:g}-{result['cap']:g}x its current budget "
        f"(grid ${result['resolution']:.2f}); gains are revenue a day"
    )
    names = list(current)
    print("  " + " " * 16 + "".join(f"{c:>14s}" for c in names) + "   projected   true gain")
    print("  " + f"{'current':16s}" + "".join(f"{current[c]:14,.0f}" for c in names))
    for name, f in result["fits"].items():
        print(
            f"  {name:16s}"
            + "".join(f"{f['spend'][c]:14,.0f}" for c in names)
            + f"  {f['projected_gain']:+9,.0f}  {f['true_gain']:+9,.0f}"
        )
    t = result["true_optimum"]
    print(
        f"  {'true optimum':16s}"
        + "".join(f"{t['spend'][c]:14,.0f}" for c in names)
        + f"  {'':9s}  {t['true_gain']:+9,.0f}"
    )
    print(
        "  fits moving each channel up / down: "
        + ", ".join(f"{c} {v['up']}/{v['down']}" for c, v in s["channels"].items())
    )
    pg, tg = s["projected_gain"], s["true_gain"]
    print(
        f"  projected gain ${pg['min']:,.0f} to ${pg['max']:,.0f} a day (median "
        f"${pg['median']:,.0f}); true gain ${tg['min']:,.0f} to ${tg['max']:,.0f} "
        f"(median ${tg['median']:,.0f}), below zero for {s['true_gain_below_zero']} of "
        f"{s['fits']} fits; at most ${t['true_gain']:,.0f}"
    )
    print(
        f"  every gain is over the current budgets spent steadily; that alone earns "
        f"${t['steady_gain']:,.0f} a day more than they did as really spent"
    )


def _job(seed: int, kwargs: dict) -> dict:
    return plan(seed, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m marketing_roi.optimise",
        description="Reallocate the current budget over the fitted response curves, under "
        "every fit, and show how far the answer moves.",
    )
    parser.add_argument(
        "--seeds",
        type=identify.seed_list,
        default=[42],
        help="generated runs, e.g. 42 (data/generated) or 20-29",
    )
    parser.add_argument("--workers", type=int, default=1, help="runs fitted at once")
    parser.add_argument("--fits", default=",".join(FITS), help="comma-separated")
    parser.add_argument("--floor", type=float, default=0.5, help="least budget, x current")
    parser.add_argument("--cap", type=float, default=2.0, help="most budget, x current")
    parser.add_argument("--holdout-days", type=int, default=91)
    parser.add_argument("--steps", type=int, default=4000, help="grid parts of the total")
    parser.add_argument("--days", type=int, default=731, help="days generated")
    parser.add_argument(
        "--truth",
        choices=tuple(TRUTHS),
        default="default",
        help="the generator's curves, or every channel's set away from the priors' centre",
    )
    parser.add_argument("--json", type=Path, help="also write the figures")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    started = time.perf_counter()
    kwargs = {
        "fits": tuple(f for f in args.fits.split(",") if f),
        "floor": args.floor,
        "cap": args.cap,
        "holdout_days": args.holdout_days,
        "steps": args.steps,
        "days": args.days,
        "truth": args.truth,
    }
    try:
        if args.workers == 1 or len(args.seeds) == 1:
            results = [plan(seed, **kwargs) for seed in args.seeds]
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                results = list(pool.map(_job, args.seeds, [kwargs] * len(args.seeds)))
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    for result in results:
        _print_run(result)
    summary = summarise(results)
    if len(results) > 1:
        print(
            f"over {len(results)} runs, medians as a share of new-customer revenue: "
            "projected / true gain, runs with a true loss, share of the best possible gain"
        )
        for name, v in summary.items():
            if name == "true_optimum":
                print(f"  {'true optimum':16s} {'':>7s}  {v['true_share']:+6.1%}")
                print(
                    f"  spending the current budgets steadily, before any reallocation: "
                    f"{v['steady_share']:+.1%}"
                )
                continue
            captured = "n/a" if v["captured"] is None else f"{v['captured']:.0%}"
            print(
                f"  {name:16s} {v['projected_share']:+6.1%}  {v['true_share']:+6.1%}"
                f"  {v['true_below_zero']:2d}  {captured}"
            )
    print(f"  {time.perf_counter() - started:.0f} s")
    if args.json:
        figures = {"runs": results, "summary": summary}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(figures, indent=2, allow_nan=False) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
