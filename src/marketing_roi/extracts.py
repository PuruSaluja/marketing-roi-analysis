"""Flat CSV extracts of the analysis views and the optimiser's results, for Tableau.

``python -m marketing_roi.extracts --out extracts`` writes one file per table below.
Each has one header row, dates as ISO ``YYYY-MM-DD``, booleans as ``true``/``false``
and an empty field for SQL's NULL, which Tableau's text connector reads as typed
columns. The views' ratios are left out wherever their parts are in the file:
a dashboard has to take a ratio over sums (``SUM([spend]) / SUM([clicks])``), and a
column of per-row ratios invites averaging them.

From the ``analysis`` views (last-click figures, as a business would see them):

``channel_daily.csv``
    One row per day and paid channel: spend, impressions, clicks, and the sessions,
    purchases and revenue GA4 credits to the channel.
``acquisition_monthly.csv``
    One row per month and channel group: spend and the new customers whose first
    purchase is credited to the group.
``blended_monthly.csv``
    One row per month: all spend against all revenue and new customers.
``cohorts.csv``
    One row per first-purchase month and month after it, with the cohort's purchases
    and revenue that month and cumulatively.
``ltv_by_channel.csv``
    ``analysis.ltv_by_channel(window)``, one row per channel group: figures per
    customer, so not to be added across groups.

From one run of the optimiser (``python -m marketing_roi.optimise --json``), if
``--optimise`` names its file; the run must be of the loaded data, which is checked
against its current budgets:

``budget_splits.csv``
    One row per split and channel: its daily budget against the current one.
``budget_gains.csv``
    One row per split: the gain its own fit projects and the true gain, per day over
    the current budgets spent steadily.
``holdout_lift.csv``
    One row per split, applied to the held-out quarter: the lift each fit projects
    (its own, the least and the most over all fits) and the true lift.

The true gains and lifts come from the generator's curves, which only generated data
has. They are labelled ``true_`` and are never to be shown as a measured result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

LTV_DAYS = 180

# Every file's columns, with the type the dashboard reads each as.
COLUMNS: dict[str, dict[str, str]] = {
    "channel_daily": {
        "date": "date", "month": "date", "channel": "text", "channel_group": "text",
        "spend": "number", "impressions": "number", "clicks": "number",
        "sessions": "number", "purchases": "number", "revenue": "number",
    },
    "acquisition_monthly": {
        "month": "date", "channel_group": "text", "paid": "bool", "spend": "number",
        "new_customers": "number",
    },
    "blended_monthly": {
        "month": "date", "spend": "number", "purchases": "number", "revenue": "number",
        "new_customers": "number", "new_customer_revenue": "number",
        "paid_purchases": "number", "paid_revenue": "number",
    },
    "cohorts": {
        "cohort_month": "date", "months_since": "number", "month": "date",
        "cohort_size": "number", "active_customers": "number", "purchases": "number",
        "revenue": "number", "cumulative_purchases": "number",
        "cumulative_revenue": "number", "complete": "bool",
    },
    "ltv_by_channel": {
        "window_days": "number", "channel_group": "text", "paid": "bool",
        "customers": "number", "spend": "number", "cac": "number",
        "revenue_per_customer": "number", "ltv_to_cac": "number",
    },
    "budget_splits": {
        "split": "text", "kind": "text", "channel": "text", "channel_group": "text",
        "spend_per_day": "number", "current_per_day": "number",
    },
    "budget_gains": {
        "split": "text", "kind": "text", "projected_gain_per_day": "number",
        "projected_gain_share": "number", "true_gain_per_day": "number",
        "true_gain_share": "number", "new_customer_revenue_per_day": "number",
        "holdout_mape": "number",
    },
    "holdout_lift": {
        "split": "text", "kind": "text", "first_day": "date", "days": "number",
        "quarter_spend": "number", "quarter_new_customer_revenue": "number",
        "own_projected_lift": "number", "least_projected_lift": "number",
        "most_projected_lift": "number", "true_lift": "number",
    },
}  # fmt: skip
VIEW_FILES = ("channel_daily", "acquisition_monthly", "blended_monthly", "cohorts",
              "ltv_by_channel")  # fmt: skip
OPTIMISER_FILES = ("budget_splits", "budget_gains", "holdout_lift")


class ExtractError(Exception):
    """The extracts cannot be read, written or matched to the loaded data."""


def read_views(conninfo: str, *, ltv_days: int = LTV_DAYS) -> dict[str, list[dict]]:
    """The rows of the five view files, from the database at ``conninfo``."""
    if isinstance(ltv_days, bool) or not isinstance(ltv_days, int) or ltv_days < 1:
        raise ExtractError(f"ltv_days must be a whole number of days, at least 1: {ltv_days!r}")
    try:
        with psycopg.connect(conninfo, row_factory=dict_row) as conn:
            out = {
                "channel_daily": conn.execute(
                    "SELECT date, date_trunc('month', date)::date AS month, channel, "
                    "channel_group, spend, impressions, clicks, sessions, purchases, revenue "
                    "FROM analysis.channel_daily ORDER BY date, channel"
                ).fetchall(),
                "acquisition_monthly": conn.execute(
                    "SELECT month, channel_group, paid, spend, new_customers "
                    "FROM analysis.acquisition WHERE month IS NOT NULL "
                    "ORDER BY month, channel_group"
                ).fetchall(),
                "blended_monthly": conn.execute(
                    "SELECT month, spend, purchases, revenue, new_customers, "
                    "new_customer_revenue, paid_purchases, paid_revenue "
                    "FROM analysis.blended WHERE month IS NOT NULL ORDER BY month"
                ).fetchall(),
                "cohorts": conn.execute(
                    "SELECT cohort_month, months_since, month, cohort_size, active_customers, "
                    "purchases, revenue, "
                    "sum(purchases) OVER w AS cumulative_purchases, "
                    "sum(revenue) OVER w AS cumulative_revenue, complete "
                    "FROM analysis.cohorts "
                    "WINDOW w AS (PARTITION BY cohort_month ORDER BY months_since) "
                    "ORDER BY cohort_month, months_since"
                ).fetchall(),
                "ltv_by_channel": conn.execute(
                    "SELECT %s AS window_days, * FROM analysis.ltv_by_channel(%s)",
                    (ltv_days, ltv_days),
                ).fetchall(),
            }
    except psycopg.Error as exc:
        raise ExtractError(f"cannot read the analysis views: {exc}".strip()) from exc
    if not out["channel_daily"]:
        raise ExtractError("analysis.channel_daily has no rows: load the data first")
    return out


def read_optimiser(path: Path, seed: int | None = None) -> dict:
    """One run from an optimiser results file: the only one, or the one with ``seed``."""
    try:
        runs = json.loads(Path(path).read_text(encoding="utf-8"))["runs"]
    except OSError as exc:
        raise ExtractError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise ExtractError(f"{path} is not an optimiser results file: {exc}") from exc
    if not isinstance(runs, list) or not runs:
        raise ExtractError(f"{path} holds no runs")
    if seed is None:
        if len(runs) > 1:
            seeds = ", ".join(str(r.get("seed")) for r in runs)
            raise ExtractError(
                f"{path} holds {len(runs)} runs (seeds {seeds}): pick one with --seed"
            )
        return runs[0]
    picked = [r for r in runs if r.get("seed") == seed]
    if not picked:
        raise ExtractError(f"{path} has no run with seed {seed}")
    return picked[0]


def check_run(run: Mapping, daily: list[dict]) -> None:
    """Refuse a run whose current budgets are not the loaded data's: each channel's mean
    daily spend over the run's training days."""
    try:
        n, current = int(run["train_days"]), dict(run["current"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExtractError(f"the optimiser run has no training days or budgets: {exc}") from exc
    days = sorted({r["date"] for r in daily})[:n]
    if len(days) < n:
        raise ExtractError(f"the run trained on {n} days; the loaded data has {len(days)}")
    last = days[-1]
    spend: dict[str, float] = {}
    for r in daily:
        if r["date"] <= last:
            spend[r["channel"]] = spend.get(r["channel"], 0.0) + float(r["spend"])
    if set(spend) != set(current):
        raise ExtractError(
            f"the run's channels {sorted(current)} are not the loaded data's {sorted(spend)}"
        )
    for c, total in spend.items():
        if not math.isclose(total / n, float(current[c]), rel_tol=1e-9, abs_tol=1e-9):
            raise ExtractError(
                f"the run is not of the loaded data: its current {c} budget is "
                f"${float(current[c]):,.2f} a day, the data's ${total / n:,.2f}"
            )


def _kind(name: str) -> str:
    return {"current": "current", "recommendation": "recommendation",
            "true_optimum": "true optimum"}.get(name, "fit")  # fmt: skip


def optimiser_rows(run: Mapping, groups: Mapping[str, str]) -> dict[str, list[dict]]:
    """The three optimiser files' rows from one run; ``groups`` maps each channel to its
    channel group, for labels."""
    current = run["current"]
    revenue = run["new_customer_revenue"]
    splits = {"current": dict(current)}
    gains = [dict(split="current", kind="current", projected_gain_per_day=0.0,
                  projected_gain_share=0.0, true_gain_per_day=0.0, true_gain_share=0.0,
                  new_customer_revenue_per_day=revenue, holdout_mape=None)]  # fmt: skip
    for name, f in run["fits"].items():
        splits[name] = f["spend"]
        gains.append(dict(split=name, kind="fit", projected_gain_per_day=f["projected_gain"],
                          projected_gain_share=f["projected_gain_share"],
                          true_gain_per_day=f["true_gain"], true_gain_share=f["true_gain_share"],
                          new_customer_revenue_per_day=revenue,
                          holdout_mape=f.get("holdout_mape")))  # fmt: skip
    rec = run["recommendation"]
    splits["recommendation"] = rec["spend"]
    worst = run["robust"]["worst_projected_gain"] if rec["change"] else 0.0
    gains.append(dict(split="recommendation", kind="recommendation",
                      projected_gain_per_day=worst, projected_gain_share=worst / revenue,
                      true_gain_per_day=rec["true_gain"], true_gain_share=rec["true_gain_share"],
                      new_customer_revenue_per_day=revenue, holdout_mape=None))  # fmt: skip
    best = run["true_optimum"]
    splits["true_optimum"] = best["spend"]
    gains.append(dict(split="true_optimum", kind="true optimum", projected_gain_per_day=None,
                      projected_gain_share=None, true_gain_per_day=best["true_gain"],
                      true_gain_share=best["true_gain_share"],
                      new_customer_revenue_per_day=revenue, holdout_mape=None))  # fmt: skip

    split_rows = [
        dict(split=name, kind=_kind(name), channel=c, channel_group=groups.get(c, c),
             spend_per_day=spend[c], current_per_day=current[c])
        for name, spend in splits.items() for c in current
    ]  # fmt: skip

    hold = run["holdout"]
    lift_rows = []
    for name, s in hold["splits"].items():
        projected = [v for v in s["projected"].values() if v is not None]
        own = s["projected"].get(name)
        if name == "recommendation":
            own = min(projected) if projected else None
        lift_rows.append(dict(
            split=name, kind=_kind(name), first_day=dt.date.fromisoformat(hold["first_day"]),
            days=hold["days"], quarter_spend=hold["spend"],
            quarter_new_customer_revenue=hold["new_customer_revenue"],
            own_projected_lift=own,
            least_projected_lift=min(projected) if projected else None,
            most_projected_lift=max(projected) if projected else None,
            true_lift=s["true"],
        ))  # fmt: skip
    return {"budget_splits": split_rows, "budget_gains": gains, "holdout_lift": lift_rows}


def _field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExtractError(f"a figure is {value}, which a CSV extract cannot hold")
        return repr(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[Mapping]) -> int:
    """Write ``rows`` to ``path`` with a header; the number of rows."""
    columns = list(columns)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh, lineterminator="\n")
        out.writerow(columns)
        for r in rows:
            out.writerow([_field(r[c]) for c in columns])
            n += 1
    return n


def write(tables: Mapping[str, list[dict]], out: Path) -> dict[str, int]:
    """Write each table to ``out/<name>.csv``; the rows written per file."""
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise ExtractError(f"{out} is a file, not a folder")
    try:
        out.mkdir(parents=True, exist_ok=True)
        return {name: write_csv(out / f"{name}.csv", COLUMNS[name], rows)
                for name, rows in tables.items()}  # fmt: skip
    except OSError as exc:
        raise ExtractError(f"cannot write to {out}: {exc.strerror or exc}") from exc


def extract(
    conninfo: str,
    out: Path,
    *,
    optimiser: Path | None = None,
    seed: int | None = None,
    ltv_days: int = LTV_DAYS,
) -> dict[str, int]:
    """Read the views (and an optimiser run), check they belong together, and write the
    extracts. Nothing is written if a check fails."""
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise ExtractError(f"{out} is a file, not a folder")
    run = read_optimiser(optimiser, seed) if optimiser else None
    tables = read_views(conninfo, ltv_days=ltv_days)
    if run is not None:
        check_run(run, tables["channel_daily"])
        groups = {r["channel"]: r["channel_group"] for r in tables["channel_daily"]}
        try:
            tables.update(optimiser_rows(run, groups))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ExtractError(f"{optimiser} is missing part of a run: {exc!r}") from exc
    for name, rows in tables.items():
        for r in rows:
            for c in COLUMNS[name]:
                _field(r[c])  # refuse a NaN or infinity before any file is written
    return write(tables, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m marketing_roi.extracts",
        description="Write the analysis views, and an optimiser run's results, as CSV "
        "extracts for Tableau.",
    )
    parser.add_argument("--dsn", default=os.environ.get("MARKETING_ROI_DSN"),
                        help="connection string (default: $MARKETING_ROI_DSN)")  # fmt: skip
    parser.add_argument("--out", type=Path, required=True, help="folder to write the files to")
    parser.add_argument("--optimise", type=Path,
                        help="an optimiser results file (optimise --json) to add")  # fmt: skip
    parser.add_argument("--seed", type=int, help="the run to take from a file of several")
    parser.add_argument("--ltv-days", type=int, default=LTV_DAYS,
                        help="the LTV file's window (default: 180)")  # fmt: skip
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("no database: pass --dsn or set MARKETING_ROI_DSN")
    if args.ltv_days < 1:
        parser.error("--ltv-days must be at least 1")
    if args.seed is not None and args.optimise is None:
        parser.error("--seed picks a run from --optimise's file")
    try:
        rows = extract(args.dsn, args.out, optimiser=args.optimise, seed=args.seed,
                       ltv_days=args.ltv_days)  # fmt: skip
    except ExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, n in rows.items():
        print(f"  {name}.csv: {n:,} rows")
    print(f"wrote {len(rows)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
