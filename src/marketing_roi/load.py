"""Load the generator's files into PostgreSQL.

Creates the ``marketing`` and ``truth`` schemas from ``sql/schema.sql`` and copies the
four files in with ``COPY``, all in one transaction: a load that fails part-way leaves
the database as it was. After copying it checks the result against ``truth.json``
(row counts, and revenue agreeing to the cent between sessions and the truth table)
and refuses to commit if they differ.

The connection string comes from ``--dsn`` or ``MARKETING_ROI_DSN``, in any form libpq
accepts; a password can stay in ``PGPASSWORD`` or a pgpass file instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.resources import files
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

SCHEMAS = ("marketing", "truth")
FILES = ("channel_daily.csv", "ga4_sessions.csv", "truth_daily.csv", "truth.json")
CHUNK = 1 << 20


class LoadError(Exception):
    """The files or the database are not in a state that can be loaded."""


def _sql_file(name: str) -> str:
    return (files("marketing_roi") / "sql" / name).read_text(encoding="utf-8")


def _copy_file(cur: psycopg.Cursor, statement: sql.Composable, path: Path) -> None:
    with path.open("rb") as f, cur.copy(statement) as copy:
        while chunk := f.read(CHUNK):
            copy.write(chunk)


def _read_truth(path: Path) -> dict:
    try:
        truth = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LoadError(f"{path.name} is not valid JSON: {exc}") from None
    if not isinstance(truth, dict) or not all(
        isinstance(truth.get(k), dict) for k in ("channels", "rows")
    ):
        raise LoadError(f"{path.name} has no 'channels' and 'rows' objects")
    for key in ("channel_daily", "ga4_sessions", "truth_daily"):
        count = truth["rows"].get(key)
        if isinstance(count, bool) or not isinstance(count, int):
            raise LoadError(f"{path.name}: rows.{key} must be a row count, got {count!r}")
    for name, ch in truth["channels"].items():
        if not isinstance(ch, dict):
            raise LoadError(f"{path.name}: channel {name!r} must be an object")
        missing = [k for k in ("source", "medium", "channel_group", "flighted") if k not in ch]
        if missing:
            raise LoadError(f"{path.name}: channel {name!r} lacks {', '.join(missing)}")
        if not all(isinstance(ch[k], str) for k in ("source", "medium", "channel_group")) or (
            not isinstance(ch["flighted"], bool)
        ):
            raise LoadError(
                f"{path.name}: channel {name!r} needs text source, medium and channel_group "
                "and a true/false flighted"
            )
    return truth


def load(conninfo: str, data_dir: Path, *, replace: bool = False) -> dict[str, int]:
    """Load `data_dir` into the database at `conninfo`; return rows loaded per table.

    Refuses if the schemas already exist, unless `replace` is set, which drops them
    (and everything in them) first.
    """
    data_dir = Path(data_dir)
    missing = [name for name in FILES if not (data_dir / name).is_file()]
    if missing:
        raise LoadError(f"{data_dir} lacks {', '.join(missing)}; run marketing_roi.generate first")
    truth = _read_truth(data_dir / "truth.json")

    try:
        conn = psycopg.connect(conninfo)
    except psycopg.Error as exc:
        raise LoadError(f"cannot connect to the database: {exc}".strip()) from exc
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT nspname FROM pg_namespace WHERE nspname = ANY(%s) ORDER BY 1",
                (list(SCHEMAS),),
            )
            existing = [r[0] for r in cur.fetchall()]
            if existing and not replace:
                schemas = " and ".join(existing)
                raise LoadError(
                    f"{'schemas' if len(existing) > 1 else 'schema'} {schemas} already "
                    f"{'exist' if len(existing) > 1 else 'exists'}; load with replace "
                    "(--replace) to drop and load again"
                )
            if existing:
                cur.execute("DROP SCHEMA IF EXISTS marketing, truth CASCADE")
            cur.execute(_sql_file("schema.sql"))
            counts = _copy_in(cur, data_dir, truth)
            _check(cur, truth, counts)
            cur.execute(_sql_file("indexes.sql"))
    except psycopg.Error as exc:
        # COPY reports the file's line; the rest of the message says what was wrong.
        raise LoadError(f"the database refused the load: {exc}".strip()) from exc
    # Outside the load's transaction, as VACUUM must be: statistics for the planner, and
    # the visibility map, without which an index-only scan still reads the table. The
    # load is committed by now, so a failure here must not read as a refused load.
    try:
        with psycopg.connect(conninfo, autocommit=True) as vacuum_conn:
            vacuum_conn.execute(
                "VACUUM (ANALYZE) marketing.channels, marketing.channel_daily, "
                "marketing.traffic_sources, marketing.ga4_sessions, truth.daily, truth.run"
            )
    except psycopg.Error as exc:
        raise LoadError(
            f"the data is loaded and committed, but VACUUM (ANALYZE) failed: {exc}".strip()
            + "; run it on the marketing and truth tables before querying"
        ) from exc
    return counts


def _copy_in(cur: psycopg.Cursor, data_dir: Path, truth: dict) -> dict[str, int]:
    for channel_id, (name, ch) in enumerate(truth["channels"].items(), start=1):
        cur.execute(
            "INSERT INTO marketing.channels VALUES (%s, %s, %s, %s, %s, %s)",
            (channel_id, name, ch["source"], ch["medium"], ch["channel_group"], ch["flighted"]),
        )

    # HEADER MATCH makes COPY check the header names against the staging columns.
    cur.execute("""
        CREATE TEMP TABLE stage_channel_daily (
            date date, channel text, spend numeric, impressions bigint, clicks bigint,
            conversions integer, revenue numeric
        ) ON COMMIT DROP""")
    _copy_file(
        cur,
        sql.SQL("COPY stage_channel_daily FROM STDIN (FORMAT csv, HEADER MATCH)"),
        data_dir / "channel_daily.csv",
    )
    cur.execute("""
        SELECT DISTINCT s.channel FROM stage_channel_daily s
        WHERE NOT EXISTS (SELECT FROM marketing.channels c WHERE c.name = s.channel)""")
    unknown = [r[0] for r in cur.fetchall()]
    if unknown:
        raise LoadError(f"channel_daily.csv names channels truth.json lacks: {unknown}")
    cur.execute("""
        INSERT INTO marketing.channel_daily
        SELECT s.date, c.channel_id, s.spend, s.impressions, s.clicks, s.conversions, s.revenue
        FROM stage_channel_daily s JOIN marketing.channels c ON c.name = s.channel
        ORDER BY s.date, c.channel_id""")

    cur.execute("""
        CREATE TEMP TABLE stage_sessions (
            session_id bigint, session_start timestamp, user_pseudo_id text, source text,
            medium text, campaign text, default_channel_group text, device_category text,
            engaged_session boolean, transaction_id text, purchase_revenue numeric
        ) ON COMMIT DROP""")
    _copy_file(
        cur,
        sql.SQL("COPY stage_sessions FROM STDIN (FORMAT csv, HEADER MATCH)"),
        data_dir / "ga4_sessions.csv",
    )
    cur.execute("""
        INSERT INTO marketing.traffic_sources
            (source, medium, campaign, default_channel_group, channel_id)
        SELECT d.source, d.medium, d.campaign, d.default_channel_group, c.channel_id
        FROM (SELECT DISTINCT source, medium, campaign, default_channel_group
              FROM stage_sessions) d
        LEFT JOIN marketing.channels c ON (c.source, c.medium) = (d.source, d.medium)
        ORDER BY c.channel_id NULLS LAST, d.source, d.medium, d.campaign""")
    cur.execute("""
        INSERT INTO marketing.ga4_sessions
            (session_id, session_start, user_pseudo_id, source_id, device_category,
             engaged_session, transaction_id, purchase_revenue)
        SELECT s.session_id, s.session_start, s.user_pseudo_id, t.source_id, s.device_category,
               s.engaged_session, s.transaction_id, s.purchase_revenue
        FROM stage_sessions s
        JOIN marketing.traffic_sources t USING (source, medium, campaign)
        ORDER BY s.session_id""")

    _copy_file(
        cur,
        sql.SQL("COPY truth.daily FROM STDIN (FORMAT csv, HEADER MATCH)"),
        data_dir / "truth_daily.csv",
    )
    cur.execute("INSERT INTO truth.run (params) VALUES (%s)", (Jsonb(truth),))

    counts = {}
    for table in (
        "marketing.channel_daily",
        "marketing.traffic_sources",
        "marketing.ga4_sessions",
        "truth.daily",
    ):
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(*table.split("."))))
        counts[table] = cur.fetchone()[0]
    return counts


def _check(cur: psycopg.Cursor, truth: dict, counts: dict[str, int]) -> None:
    rows = truth["rows"]
    for key, table in (
        ("channel_daily", "marketing.channel_daily"),
        ("ga4_sessions", "marketing.ga4_sessions"),
        ("truth_daily", "truth.daily"),
    ):
        if rows[key] != counts[table]:
            raise LoadError(f"{table} has {counts[table]:,} rows; truth.json says {rows[key]:,}")
    # Day by day, so edits that cancel out across days are caught too.
    cur.execute("""
        WITH s AS (
            SELECT session_date AS date, count(*) AS conversions, sum(purchase_revenue) AS revenue
            FROM marketing.ga4_sessions WHERE transaction_id IS NOT NULL GROUP BY 1),
        t AS (
            SELECT date, sum(conversions) AS conversions, sum(revenue) AS revenue
            FROM truth.daily GROUP BY 1 HAVING sum(conversions) > 0)
        SELECT date, coalesce(s.conversions, 0), coalesce(s.revenue, 0),
               coalesce(t.conversions, 0), coalesce(t.revenue, 0)
        FROM s FULL JOIN t USING (date)
        WHERE (coalesce(s.conversions, 0), coalesce(s.revenue, 0))
              IS DISTINCT FROM (coalesce(t.conversions, 0), coalesce(t.revenue, 0))
        ORDER BY date LIMIT 1""")
    bad = cur.fetchone()
    if bad:
        date, conv, rev, t_conv, t_rev = bad
        raise LoadError(
            f"on {date} the sessions hold {conv} purchases worth {rev}; "
            f"truth.daily says {t_conv} worth {t_rev}"
        )
    cur.execute("""
        SELECT d.date, c.name, d.conversions, d.revenue, t.conversions, t.revenue
        FROM marketing.channel_daily d
        JOIN marketing.channels c USING (channel_id)
        LEFT JOIN truth.daily t ON (t.date, t.component) = (d.date, c.name)
        WHERE (d.conversions, d.revenue) IS DISTINCT FROM (t.conversions, t.revenue)
        ORDER BY d.date, c.name LIMIT 1""")
    bad = cur.fetchone()
    if bad:
        date, name, conv, rev, t_conv, t_rev = bad
        raise LoadError(
            f"channel_daily has {conv} purchases worth {rev} for {name} on {date}; "
            f"truth.daily says {t_conv} worth {t_rev}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m marketing_roi.load",
        description="Load the generated files into PostgreSQL (see the module docstring).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/generated"),
        help="folder the generator wrote (default: data/generated)",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("MARKETING_ROI_DSN"),
        help="connection string (default: $MARKETING_ROI_DSN)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="drop the marketing and truth schemas first if they exist",
    )
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("no database: pass --dsn or set MARKETING_ROI_DSN")
    started = time.perf_counter()
    try:
        counts = load(args.dsn, args.data, replace=args.replace)
    except LoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"loaded in {time.perf_counter() - started:.1f} s")
    for table, n in counts.items():
        print(f"  {table:28s} {n:>10,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
