"""Does the mix model recover each channel's curve? Fits on generated runs, many seeds.

Held-out accuracy says little about the per-channel curves: the fit can match the
held-out quarter's totals while crediting a channel with a fraction, or a multiple, of
the revenue it caused. A budget optimiser runs on the curves, so this checks them
directly against the generator's truth, two ways per channel:

``credit``
    The revenue the fit credits to the channel over the training days, over the revenue
    the channel caused there (truth_daily; 1 is right).
``marginal``
    The fitted curve's extra revenue a day from one more dollar a day, at a steady spend
    equal to the channel's mean training spend, over the true curve's (the true purchase
    curve times the mean order value; 1 is right). This is the figure a reallocation
    equalises across channels.

Each run is fitted several ways (``VARIANTS``): as committed, with priors on the curves'
shapes and half-saturations, on weekly sums, with a measured lift for paid search, and
given the true curves (the regression's ceiling). The lift stands in for a holdout
experiment: over a 28-day window of the training days its "measured" revenue is the
revenue paid search really caused there, times 1 + e with e drawn from N(0, 0.15), and
its stated standard error is 15% of that; the draw is seeded by the run's seed.

``--moved-truth`` generates the runs with every channel's true shape at 2.5 and
half-saturation at 2.5 times its daily spend, away from the priors' centre, to show
whether a prior finds the truth or only its own centre.

    python -m marketing_roi.identify --seeds 10-19 --json results/identify-dev-10-19.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from marketing_roi import mmm
from marketing_roi.generate import GeneratorConfig, generate
from marketing_roi.saturation import true_curve

WITHIN = 0.25
"""A ratio within 25% of 1 counts as recovered."""
LIFT_CHANNEL = "paid_search"
LIFT_START_DAY = 182
"""The lift window's first day, counted from the first day (2024-07-01 by default)."""
LIFT_DAYS = 28
LIFT_SE = 0.15

VARIANTS = (
    "committed",
    "prior_0.25",
    "prior_0.5",
    "prior_1",
    "weekly",
    "lift",
    "prior_0.25+lift",
    "prior_0.5+lift",
    "prior_1+lift",
    "true_curves",
)


def moved(config: GeneratorConfig) -> GeneratorConfig:
    """The same run with every channel's true curve away from the priors' centre."""
    channels = tuple(
        dataclasses.replace(c, hill_shape=2.5, half_saturation=2.5) for c in config.channels
    )
    return dataclasses.replace(config, channels=channels)


def lift_for(data, caused: dict[str, np.ndarray]) -> mmm.Lift:
    """Paid search's "measured" lift over its window, as a holdout experiment might
    report it."""
    rng = np.random.default_rng([data.config.seed, 2])
    window = slice(LIFT_START_DAY, LIFT_START_DAY + LIFT_DAYS)
    true = float(caused[LIFT_CHANNEL][window].sum())
    return mmm.Lift(
        LIFT_CHANNEL,
        data.dates[window.start],
        data.dates[window.stop - 1],
        revenue=true * (1 + rng.normal(0, LIFT_SE)),
        se=LIFT_SE * true,
    )


def caused_revenue(data) -> dict[str, np.ndarray]:
    """Each channel's caused revenue a day, in dollars, from the run's truth."""
    td = data.truth_daily
    n = data.config.days
    out = {}
    for c, ch in enumerate(data.config.channels):
        rows = td["component"] == c
        out[ch.name] = np.bincount(td["day"][rows], weights=td["revenue_cents"][rows], minlength=n)
        out[ch.name] = out[ch.name] / 100
    return out


def options(variant: str, data, caused, holdout_days: int) -> dict:
    out: dict = {"holdout_days": holdout_days}
    for part in variant.split("+"):
        if part.startswith("prior_"):
            out["prior"] = mmm.Prior(sd=float(part.removeprefix("prior_")))
        elif part == "weekly":
            out["weekly"] = True
        elif part == "lift":
            out["lifts"] = (lift_for(data, caused),)
        elif part == "true_curves":
            out["fixed"] = {
                c.name: (c.decay, c.half_saturation * c.daily_spend, c.hill_shape)
                for c in data.config.channels
            }
        elif part != "committed":
            raise ValueError(f"unknown variant {variant!r}")
    return out


def run_seed(
    seed: int, variants: tuple[str, ...], move: bool, holdout_days: int, days: int = 731
) -> dict:
    """Every variant's fit on one generated run, scored against the truth."""
    config = GeneratorConfig(seed=seed, days=days)
    if any("lift" in v.split("+") for v in variants) and (
        days - holdout_days < LIFT_START_DAY + LIFT_DAYS
    ):
        raise ValueError(
            f"the lift's window (days {LIFT_START_DAY} to {LIFT_START_DAY + LIFT_DAYS - 1}) "
            f"needs at least {LIFT_START_DAY + LIFT_DAYS} training days"
        )
    data = generate(moved(config) if move else config)
    inputs = mmm.inputs_from_dataset(data)
    caused = caused_revenue(data)
    aov = data.config.aov
    out: dict = {"seed": seed, "variants": {}}
    for variant in variants:
        model = mmm.fit(inputs, **options(variant, data, caused, holdout_days))
        n = model.train_days
        parts = model.decompose(inputs)
        channels = {}
        for c, (ch, fitted) in enumerate(zip(data.config.channels, model.channels, strict=True)):
            x = float(inputs.spend[c, :n].mean())
            truth = true_curve(ch, data.betas[ch.name])
            true_marginal = float(truth.marginal(x)) * aov
            fit_marginal = float(fitted.curve.marginal(x))
            channels[ch.name] = {
                "credit": float(parts[ch.name][:n].sum()) / float(caused[ch.name][:n].sum()),
                "marginal": fit_marginal / true_marginal,
                "marginal_fitted": fit_marginal,
                "marginal_true": true_marginal,
                "decay": fitted.decay,
                "half_saturation": fitted.half_saturation / x,
                "shape": fitted.shape,
            }
        scores = mmm.evaluate(model, inputs)
        out["variants"][variant] = {
            "channels": channels,
            "holdout_mape": scores["holdout"]["mape"] if holdout_days else None,
            "holdout_r2": scores["holdout"]["r2"] if holdout_days else None,
            "seconds": round(model.seconds, 1),
        }
    return out


def within(ratio: float) -> bool:
    return abs(ratio - 1) <= WITHIN


def summarise(runs: list[dict], variants: tuple[str, ...]) -> dict:
    """Per variant and channel, on how many seeds the credit, the marginal, and both
    came within 25%; and the median held-out MAPE."""
    out = {}
    for v in variants:
        rows = [r["variants"][v] for r in runs]
        names = list(rows[0]["channels"])
        per = {}
        for name in names:
            got = [row["channels"][name] for row in rows]
            per[name] = {
                "credit_within": sum(within(g["credit"]) for g in got),
                "marginal_within": sum(within(g["marginal"]) for g in got),
                "both_within": sum(within(g["credit"]) and within(g["marginal"]) for g in got),
                "credit_range": [min(g["credit"] for g in got), max(g["credit"] for g in got)],
                "marginal_range": [
                    min(g["marginal"] for g in got),
                    max(g["marginal"] for g in got),
                ],
            }
        mapes = [row["holdout_mape"] for row in rows if row["holdout_mape"] is not None]
        out[v] = {
            "seeds": len(rows),
            "channels": per,
            "both_within_pairs": sum(p["both_within"] for p in per.values()),
            "median_holdout_mape": float(np.median(mapes)) if mapes else None,
        }
    return out


def seed_list(text: str) -> list[int]:
    """'10-19' or '3,5,8' (or a mix) to the seeds it names."""
    seeds: list[int] = []
    for part in text.split(","):
        a, sep, b = part.partition("-")
        lo, hi = int(a), int(b) if sep else int(a)
        if hi < lo:
            raise argparse.ArgumentTypeError(f"{part!r} runs backwards")
        seeds.extend(range(lo, hi + 1))
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("a seed is named twice")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m marketing_roi.identify",
        description="Fit the mix model several ways on generated runs and check each "
        "channel's fitted curve against the truth.",
    )
    parser.add_argument("--seeds", type=seed_list, required=True, help="e.g. 10-19")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated")
    parser.add_argument("--moved-truth", action="store_true", help="truth away from the prior")
    parser.add_argument("--holdout-days", type=int, default=91)
    parser.add_argument("--days", type=int, default=731, help="days generated per run")
    parser.add_argument("--workers", type=int, default=1, help="seeds fitted at once")
    parser.add_argument("--json", type=Path, help="also write every figure to this file")
    args = parser.parse_args(argv)
    variants = tuple(v for v in args.variants.split(",") if v)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown or not variants:
        parser.error(f"unknown variants: {unknown}; choose from {', '.join(VARIANTS)}")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    started = time.perf_counter()
    jobs = [(s, variants, args.moved_truth, args.holdout_days, args.days) for s in args.seeds]
    try:
        if args.workers == 1:
            runs = [run_seed(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                runs = list(pool.map(run_seed, *zip(*jobs, strict=True)))
    except ValueError as exc:
        parser.error(str(exc))
    summary = summarise(runs, variants)
    print(
        f"seeds {args.seeds[0]}-{args.seeds[-1]} ({len(runs)}), "
        f"{'moved truth' if args.moved_truth else 'default truth'}: seeds with credit / "
        f"marginal / both within {WITHIN:.0%}"
    )
    for v in variants:
        s = summary[v]
        cells = "  ".join(
            f"{name} {p['credit_within']}/{p['marginal_within']}/{p['both_within']}"
            for name, p in s["channels"].items()
        )
        mape = s["median_holdout_mape"]
        print(
            f"  {v:16s} {cells}  pairs {s['both_within_pairs']:2d}"
            f"  held-out MAPE {'n/a' if mape is None else f'{mape:.1%}'}"
        )
    print(f"  {time.perf_counter() - started:.0f} s")
    if args.json:
        figures = {
            "seeds": args.seeds,
            "moved_truth": args.moved_truth,
            "holdout_days": args.holdout_days,
            "days": args.days,
            "within": WITHIN,
            "lift": {
                "channel": LIFT_CHANNEL,
                "start_day": LIFT_START_DAY,
                "days": LIFT_DAYS,
                "se": LIFT_SE,
            },
            "summary": summary,
            "runs": runs,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(figures, indent=2, allow_nan=False) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
