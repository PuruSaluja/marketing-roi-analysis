"""Channel performance, CAC, blended ROI and cohorts, as views in PostgreSQL.

``sql/analysis.sql`` creates the ``analysis`` schema over the loaded data:

``analysis.channel_performance``
    Spend, impressions, clicks, sessions, last-click purchases and revenue per paid
    channel, by month and over the whole period, with CTR, CPC, conversion rate, cost
    per purchase and ROAS.
``analysis.acquisition``
    New customers by the channel group their first purchase is credited to, and each
    paid group's customer acquisition cost.
``analysis.blended``
    All spend against all revenue and all new customers: blended ROAS, ROI on revenue
    and CAC.
``analysis.cohorts``
    Customers by the month of their first purchase, followed month by month: how many
    buy again, and cumulative revenue per customer.
``analysis.ltv_by_channel(window_days)``
    Revenue per customer over their first ``window_days`` days, by acquiring channel
    group, against that group's CAC.

The views read only what a business would see (platform spend and clicks, GA4
sessions), never the ``truth`` schema, so every purchase is credited by last click.

``python -m marketing_roi.analysis`` (re)creates the views and prints the period's
figures. The loader creates them too, at the end of its transaction.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.resources import files

import psycopg
from psycopg.rows import dict_row

SCHEMA = "analysis"

# Everything sql/analysis.sql creates in the schema, as pg_identify_object names it.
OBJECTS = frozenset(
    [f"view analysis.{name}" for name in (
        "acquisition", "blended", "channel_daily", "channel_performance", "cohorts",
        "customers", "period", "purchases",
    )]
    + ["function analysis.ltv_by_channel(integer)"]
)  # fmt: skip


class AnalysisError(Exception):
    """The analysis views cannot be created or read."""


def _objects(cur: psycopg.Cursor) -> list[str]:
    """Everything in the schema that DROP SCHEMA ... CASCADE would drop with it."""
    cur.execute(
        """
        SELECT o.type || ' ' || o.identity
        FROM pg_depend d
        JOIN pg_namespace n ON n.oid = d.refobjid
        CROSS JOIN LATERAL pg_identify_object(d.classid, d.objid, d.objsubid) o
        WHERE d.refclassid = 'pg_namespace'::regclass AND n.nspname = %s
        ORDER BY 1""",
        (SCHEMA,),
    )
    return [r[0] for r in cur.fetchall()]


def install(cur: psycopg.Cursor) -> None:
    """Drop and recreate the ``analysis`` schema's views, in the cursor's transaction.

    Refuses if the schema holds anything this module did not create (a table, a view
    or function of someone's own, a type), since dropping the schema would drop that
    too.
    """
    other = [o for o in _objects(cur) if o not in OBJECTS]
    if other:
        raise AnalysisError(
            f"schema {SCHEMA} holds more than the analysis views ({', '.join(other)}); "
            "it is dropped and recreated, so move them elsewhere first"
        )
    cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    cur.execute((files("marketing_roi") / "sql" / "analysis.sql").read_text(encoding="utf-8"))


def create_views(conninfo: str) -> None:
    """Create the analysis views in the database at `conninfo`, in one transaction."""
    try:
        with psycopg.connect(conninfo) as conn, conn.cursor() as cur:
            install(cur)
    except psycopg.Error as exc:
        raise AnalysisError(f"cannot create the analysis views: {exc}".strip()) from exc


def report(conninfo: str, *, ltv_days: int = 180) -> dict[str, list[dict]]:
    """The whole period's figures from the analysis views, as lists of rows."""
    if isinstance(ltv_days, bool) or not isinstance(ltv_days, int) or ltv_days < 1:
        raise AnalysisError(f"ltv_days must be a whole number of days, at least 1: {ltv_days!r}")
    try:
        with psycopg.connect(conninfo, row_factory=dict_row) as conn:
            period = conn.execute("SELECT * FROM analysis.period").fetchone()
            return {
                "period": [period],
                "channels": conn.execute(
                    "SELECT * FROM analysis.channel_performance WHERE month IS NULL "
                    "ORDER BY spend DESC, channel"
                ).fetchall(),
                "acquisition": conn.execute(
                    "SELECT * FROM analysis.acquisition WHERE month IS NULL "
                    "ORDER BY paid DESC, new_customers DESC, channel_group"
                ).fetchall(),
                "blended": conn.execute(
                    "SELECT * FROM analysis.blended WHERE month IS NULL"
                ).fetchall(),
                "ltv": conn.execute(
                    "SELECT * FROM analysis.ltv_by_channel(%s)", (ltv_days,)
                ).fetchall(),
                "retention": conn.execute(
                    """
                    SELECT months_since,
                           sum(active_customers)::numeric / sum(cohort_size) AS retention,
                           sum(cohort_size) AS customers
                    FROM analysis.cohorts WHERE complete AND months_since BETWEEN 1 AND 12
                    GROUP BY 1 ORDER BY 1"""
                ).fetchall(),
            }
    except psycopg.Error as exc:
        raise AnalysisError(f"cannot read the analysis views: {exc}".strip()) from exc


def _num(value, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def _usd(value, width: int) -> str:
    return ("n/a" if value is None else f"${value:,.2f}").rjust(width)


def _print_report(r: dict[str, list[dict]], ltv_days: int) -> None:
    p = r["period"][0]
    print(f"period {p['first_day']} to {p['last_day']}; purchases credited by last click")
    print(
        f"\n{'channel':13s} {'spend':>13s} {'CTR':>7s} {'CPC':>7s} {'conv/click':>10s}"
        f" {'purchases':>9s} {'cost/purch':>10s} {'ROAS':>5s}"
    )
    for c in r["channels"]:
        print(
            f"{c['channel']:13s} {_usd(c['spend'], 13)} {_num(c['ctr'], '7.2%')}"
            f" {_usd(c['cpc'], 7)} {_num(c['conversion_rate'], '10.2%')}"
            f" {c['purchases']:>9,} {_usd(c['cost_per_purchase'], 10)}"
            f" {_num(c['roas'], '5.2f')}"
        )
    print(f"\n{'acquired via':15s} {'new customers':>13s} {'CAC':>10s}")
    for a in r["acquisition"]:
        print(f"{a['channel_group']:15s} {a['new_customers']:>13,} {_usd(a['cac'], 10)}")
    b = r["blended"][0]
    print(
        f"\nblended: spend ${b['spend']:,.2f}, revenue ${b['revenue']:,.2f}, "
        f"ROAS {_num(b['blended_roas'], '.2f')}, ROI on revenue {_num(b['blended_roi'], '.1%')}, "
        f"CAC ${_num(b['blended_cac'], '.2f')} ({b['new_customers']:,} new customers)"
    )
    print(
        f"\n{f'first {ltv_days} days':15s} {'customers':>9s} {'revenue/customer':>16s}"
        f" {'CAC':>10s} {'LTV:CAC':>7s}"
    )
    for v in r["ltv"]:
        print(
            f"{v['channel_group']:15s} {v['customers']:>9,} {_usd(v['revenue_per_customer'], 16)}"
            f" {_usd(v['cac'], 10)} {_num(v['ltv_to_cac'], '7.2f'):>7s}"
        )
    if r["retention"]:
        print("\nmonths after first purchase: share of customers buying, complete months only")
        print(
            "  " + "  ".join(f"{x['months_since']}: {x['retention']:.1%}" for x in r["retention"])
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m marketing_roi.analysis",
        description="Create the analysis views and print the period's figures.",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("MARKETING_ROI_DSN"),
        help="connection string (default: $MARKETING_ROI_DSN)",
    )
    parser.add_argument(
        "--ltv-days",
        type=int,
        default=180,
        help="days of revenue counted per customer (default: 180)",
    )
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("no database: pass --dsn or set MARKETING_ROI_DSN")
    if args.ltv_days < 1:
        parser.error("--ltv-days must be at least 1")
    try:
        create_views(args.dsn)
        figures = report(args.dsn, ltv_days=args.ltv_days)
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_report(figures, args.ltv_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
