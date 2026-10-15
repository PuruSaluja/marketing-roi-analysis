"""The analysis views, checked against the same figures worked out in Python from the
generated files."""

import csv
import datetime as dt
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

from marketing_roi.analysis import AnalysisError, create_views, install, main, report
from marketing_roi.generate import GeneratorConfig, generate, write
from marketing_roi.load import load

# Ends on 2025-12-30, so the last month is incomplete.
CONFIG = GeneratorConfig(seed=5, start=dt.date(2025, 10, 1), days=91)


def month_of(day: dt.date) -> dt.date:
    return day.replace(day=1)


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("generated")
    write(generate(CONFIG), out)
    return out


@pytest.fixture(scope="module")
def loaded(test_dsn, generated) -> str:
    """The test database with the files loaded and the views created, for reading only."""
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS analysis, marketing, truth CASCADE")
    load(test_dsn, generated)
    create_views(test_dsn)
    return test_dsn


def query(dsn: str, statement: str, params=()) -> list[dict]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        return conn.execute(statement, params).fetchall()


@pytest.fixture(scope="module")
def expected(generated) -> dict:
    """Every figure the views give, worked out from the CSV files without SQL."""
    truth = json.loads((generated / "truth.json").read_text(encoding="utf-8"))
    by_source = {(c["source"], c["medium"]): name for name, c in truth["channels"].items()}
    group_of = {name: c["channel_group"] for name, c in truth["channels"].items()}

    with (generated / "channel_daily.csv").open(newline="", encoding="utf-8") as f:
        daily = list(csv.DictReader(f))
    with (generated / "ga4_sessions.csv").open(newline="", encoding="utf-8") as f:
        sessions = list(csv.DictReader(f))
    last_day = max(dt.date.fromisoformat(r["date"]) for r in daily)

    funnel = defaultdict(lambda: defaultdict(Decimal))  # (month, channel) -> sums
    for r in daily:
        day = dt.date.fromisoformat(r["date"])
        for key in ((month_of(day), r["channel"]), (None, r["channel"])):
            for col in ("spend", "impressions", "clicks"):
                funnel[key][col] += Decimal(r[col])
            funnel[key]["sessions"] += 0
            funnel[key]["purchases"] += 0
            funnel[key]["revenue"] += 0

    purchases = []
    for r in sessions:
        day = dt.date.fromisoformat(r["session_start"][:10])
        channel = by_source.get((r["source"], r["medium"]))
        if channel:
            for key in ((month_of(day), channel), (None, channel)):
                funnel[key]["sessions"] += 1
        if r["transaction_id"]:
            purchases.append(
                {
                    "start": r["session_start"],
                    "id": int(r["session_id"]),
                    "day": day,
                    "user": r["user_pseudo_id"],
                    "revenue": Decimal(r["purchase_revenue"]),
                    "group": r["default_channel_group"],
                    "paid": channel is not None,
                }
            )
            if channel:
                for key in ((month_of(day), channel), (None, channel)):
                    funnel[key]["purchases"] += 1
                    funnel[key]["revenue"] += Decimal(r["purchase_revenue"])

    purchases.sort(key=lambda p: (p["user"], p["start"], p["id"]))
    first = {}
    for p in purchases:
        first.setdefault(p["user"], p)
    for p in purchases:
        p["first"] = first[p["user"]] is p

    acquisition = defaultdict(lambda: {"spend": Decimal(0), "new_customers": 0, "paid": True})
    for (month, channel), f in funnel.items():
        acquisition[(month, group_of[channel])]["spend"] += f["spend"]
    for c in first.values():
        for key in ((month_of(c["day"]), c["group"]), (None, c["group"])):
            acquisition[key]["new_customers"] += 1
            acquisition[key]["paid"] = c["paid"]

    blended = defaultdict(lambda: defaultdict(Decimal))
    for (month, _), f in funnel.items():
        blended[month]["spend"] += f["spend"]
    for p in purchases:
        for month in (month_of(p["day"]), None):
            b = blended[month]
            b["purchases"] += 1
            b["revenue"] += p["revenue"]
            b["new_customers"] += p["first"]
            b["new_customer_revenue"] += p["revenue"] if p["first"] else 0
            b["paid_purchases"] += p["paid"]
            b["paid_revenue"] += p["revenue"] if p["paid"] else 0

    cohorts = {}
    sizes = defaultdict(int)
    for c in first.values():
        sizes[month_of(c["day"])] += 1
    activity = defaultdict(lambda: {"users": set(), "purchases": 0, "revenue": Decimal(0)})
    for p in purchases:
        a = activity[(month_of(first[p["user"]]["day"]), month_of(p["day"]))]
        a["users"].add(p["user"])
        a["purchases"] += 1
        a["revenue"] += p["revenue"]
    for cohort, size in sizes.items():
        month, k, cum_rev, cum_n = cohort, 0, Decimal(0), 0
        while month <= last_day:
            a = activity.get((cohort, month), {"users": (), "purchases": 0, "revenue": Decimal(0)})
            cum_rev += a["revenue"]
            cum_n += a["purchases"]
            next_month = (month + dt.timedelta(days=32)).replace(day=1)
            cohorts[(cohort, k)] = {
                "month": month,
                "cohort_size": size,
                "active_customers": len(a["users"]),
                "purchases": a["purchases"],
                "revenue": a["revenue"],
                "retention": len(a["users"]) / size,
                "cumulative_revenue_per_customer": cum_rev / size,
                "cumulative_purchases_per_customer": cum_n / size,
                "complete": next_month - dt.timedelta(days=1) <= last_day,
            }
            month, k = next_month, k + 1

    def ltv(window: int) -> dict:
        last_acquired = last_day - dt.timedelta(days=window - 1)
        out = {}
        for c in first.values():
            if c["day"] > last_acquired:
                continue
            g = out.setdefault(c["group"], {"customers": 0, "revenue": Decimal(0)})
            g["paid"] = c["paid"]
            g["customers"] += 1
        for p in purchases:
            c = first[p["user"]]
            if c["day"] <= last_acquired and p["day"] < c["day"] + dt.timedelta(days=window):
                out[c["group"]]["revenue"] += p["revenue"]
        for r in daily:
            if dt.date.fromisoformat(r["date"]) <= last_acquired:
                g = out.get(group_of[r["channel"]])
                if g is not None:
                    g["spend"] = g.get("spend", Decimal(0)) + Decimal(r["spend"])
        return out

    return {
        "funnel": funnel,
        "acquisition": acquisition,
        "blended": blended,
        "cohorts": cohorts,
        "ltv": ltv,
        "last_day": last_day,
        "customers": len(first),
        "first": first,
    }


def close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return float(a) == pytest.approx(float(b), rel=1e-12, abs=1e-15)


def ratio(n, d):
    return None if d == 0 else Decimal(n) / Decimal(d)


# --- figures ---------------------------------------------------------------------------------


def test_channel_performance_matches_the_files(loaded, expected) -> None:
    got = query(loaded, "SELECT * FROM analysis.channel_performance")
    assert len(got) == len(expected["funnel"])
    for row in got:
        f = expected["funnel"][(row["month"], row["channel"])]
        for col in ("spend", "impressions", "clicks", "sessions", "purchases", "revenue"):
            assert row[col] == f[col], (row["month"], row["channel"], col)
        assert close(row["ctr"], ratio(f["clicks"], f["impressions"]))
        assert close(row["cpc"], ratio(f["spend"], f["clicks"]))
        assert close(row["conversion_rate"], ratio(f["purchases"], f["clicks"]))
        assert close(row["cost_per_purchase"], ratio(f["spend"], f["purchases"]))
        assert close(row["roas"], ratio(f["revenue"], f["spend"]))
    months = {r["month"] for r in got}
    assert months == {None, dt.date(2025, 10, 1), dt.date(2025, 11, 1), dt.date(2025, 12, 1)}


def test_acquisition_matches_the_files(loaded, expected) -> None:
    got = query(loaded, "SELECT * FROM analysis.acquisition")
    assert len(got) == len(expected["acquisition"])
    for row in got:
        a = expected["acquisition"][(row["month"], row["channel_group"])]
        assert row["paid"] is a["paid"]
        assert (row["spend"], row["new_customers"]) == (a["spend"], a["new_customers"])
        want = ratio(a["spend"], a["new_customers"]) if a["paid"] else None
        assert close(row["cac"], want), (row["month"], row["channel_group"])
    total = [r for r in got if r["month"] is None]
    assert sum(r["new_customers"] for r in total) == expected["customers"]
    assert {r["channel_group"] for r in total if not r["paid"]} >= {"Direct", "Organic Search"}


def test_blended_figures_match_the_files(loaded, expected) -> None:
    got = query(loaded, "SELECT * FROM analysis.blended")
    assert len(got) == len(expected["blended"]) == 4
    for row in got:
        b = expected["blended"][row["month"]]
        for col in (
            "spend",
            "purchases",
            "revenue",
            "new_customers",
            "new_customer_revenue",
            "paid_purchases",
            "paid_revenue",
        ):
            assert row[col] == b[col], (row["month"], col)
        assert close(row["blended_roas"], ratio(b["revenue"], b["spend"]))
        assert close(row["blended_roi"], ratio(b["revenue"] - b["spend"], b["spend"]))
        assert close(row["paid_roas"], ratio(b["paid_revenue"], b["spend"]))
        assert close(row["blended_cac"], ratio(b["spend"], b["new_customers"]))


def test_cohorts_match_the_files(loaded, expected) -> None:
    got = query(loaded, "SELECT * FROM analysis.cohorts ORDER BY cohort_month, months_since")
    assert len(got) == len(expected["cohorts"]) == 3 + 2 + 1
    for row in got:
        c = expected["cohorts"][(row["cohort_month"], row["months_since"])]
        for col in ("month", "cohort_size", "active_customers", "purchases", "revenue", "complete"):
            assert row[col] == c[col], (row["cohort_month"], row["months_since"], col)
        for col in (
            "retention",
            "cumulative_revenue_per_customer",
            "cumulative_purchases_per_customer",
        ):
            assert close(row[col], c[col]), (row["cohort_month"], row["months_since"], col)
    assert all(r["retention"] == 1 for r in got if r["months_since"] == 0)
    assert [r["complete"] for r in got if r["month"] == dt.date(2025, 12, 1)] == [False] * 3
    assert sum(r["cohort_size"] for r in got if r["months_since"] == 0) == expected["customers"]


def test_a_month_the_data_ends_on_the_last_day_of_is_complete(loaded) -> None:
    with psycopg.connect(loaded, row_factory=dict_row) as conn:
        conn.execute("DELETE FROM marketing.channel_daily WHERE date > '2025-11-30'")
        got = conn.execute(
            "SELECT DISTINCT month, complete FROM analysis.cohorts ORDER BY month"
        ).fetchall()
        assert got == [
            {"month": dt.date(2025, 10, 1), "complete": True},
            {"month": dt.date(2025, 11, 1), "complete": True},
        ]
        conn.rollback()


@pytest.mark.parametrize("window", [1, 30, 45, 91])
def test_ltv_by_channel_matches_the_files(loaded, expected, window) -> None:
    got = query(loaded, "SELECT * FROM analysis.ltv_by_channel(%s)", (window,))
    want = expected["ltv"](window)
    assert {r["channel_group"] for r in got} == set(want)
    for row in got:
        w = want[row["channel_group"]]
        assert (row["paid"], row["customers"], row["spend"]) == (
            w["paid"],
            w["customers"],
            w.get("spend"),
        )
        assert close(row["revenue_per_customer"], w["revenue"] / w["customers"])
        cac = None if "spend" not in w else w["spend"] / w["customers"]
        assert close(row["cac"], cac)
        lift = None if not cac else (w["revenue"] / w["customers"]) / cac
        assert close(row["ltv_to_cac"], lift)
    if window == 91:
        # Only customers acquired on the first day are followed for all 91 days.
        assert sum(r["customers"] for r in got) == sum(
            1 for c in expected["first"].values() if c["day"] == CONFIG.start
        )


def test_a_window_longer_than_the_data_gives_no_rows(loaded) -> None:
    assert query(loaded, "SELECT * FROM analysis.ltv_by_channel(92)") == []


# --- what the views read --------------------------------------------------------------------


def test_the_figures_do_not_depend_on_the_generators_truth(loaded) -> None:
    """Dropping the truth schema and blanking the caused purchases changes nothing."""
    before = report(loaded, ltv_days=30)
    with psycopg.connect(loaded) as conn:
        conn.execute("UPDATE marketing.channel_daily SET conversions = 0, revenue = 0")
        conn.execute("DROP SCHEMA truth CASCADE")
        conn.execute("SET LOCAL search_path = pg_catalog")
        for statement in (
            "SELECT * FROM analysis.channel_performance",
            "SELECT * FROM analysis.acquisition",
            "SELECT * FROM analysis.blended",
            "SELECT * FROM analysis.cohorts",
            "SELECT * FROM analysis.ltv_by_channel(30)",
        ):
            conn.execute(statement).fetchall()
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM analysis.channel_performance WHERE month IS NULL "
                "ORDER BY spend DESC, channel"
            )
            assert cur.fetchall() == before["channels"]
            cur.execute("SELECT * FROM analysis.ltv_by_channel(30)")
            assert cur.fetchall() == before["ltv"]
            cur.execute("SELECT * FROM analysis.blended WHERE month IS NULL")
            assert cur.fetchall() == before["blended"]
        conn.rollback()
    assert query(loaded, "SELECT count(*) AS n FROM truth.run") == [{"n": 1}]


def test_no_view_reads_truth_or_the_caused_columns(loaded) -> None:
    used = query(
        loaded,
        """
        SELECT DISTINCT table_schema, table_name, column_name
        FROM information_schema.view_column_usage WHERE view_schema = 'analysis'""",
    )
    assert used
    assert not [u for u in used if u["table_schema"] == "truth"]
    assert not [
        u
        for u in used
        if (u["table_name"], u["column_name"])
        in (("channel_daily", "conversions"), ("channel_daily", "revenue"))
        and u["table_schema"] == "marketing"
    ]


def test_a_dark_channel_has_no_ratios_and_no_division_error(loaded) -> None:
    with psycopg.connect(loaded, row_factory=dict_row) as conn:
        conn.execute("""
            UPDATE marketing.channel_daily SET spend = 0, impressions = 0, clicks = 0
            WHERE channel_id = (SELECT channel_id FROM marketing.channels WHERE name = 'display')
              AND date >= '2025-11-01' AND date < '2025-12-01'""")
        conn.execute("""
            DELETE FROM marketing.ga4_sessions WHERE transaction_id IS NULL AND source_id IN
                (SELECT source_id FROM marketing.traffic_sources t
                 JOIN marketing.channels c USING (channel_id) WHERE c.name = 'display')
              AND session_date >= '2025-11-01' AND session_date < '2025-12-01'""")
        row = conn.execute(
            "SELECT * FROM analysis.channel_performance "
            "WHERE channel = 'display' AND month = '2025-11-01'"
        ).fetchone()
        assert (row["spend"], row["impressions"], row["clicks"]) == (0, 0, 0)
        assert (row["ctr"], row["cpc"], row["conversion_rate"], row["roas"]) == (None,) * 4
        acquired = conn.execute(
            "SELECT * FROM analysis.acquisition "
            "WHERE channel_group = 'Display' AND month = '2025-11-01'"
        ).fetchone()
        # Customers credited to a dark channel by last click cost it nothing.
        assert acquired["cac"] == (0 if acquired["new_customers"] else None)
        conn.rollback()


# --- installing and reporting ---------------------------------------------------------------


def test_creating_the_views_again_replaces_them(loaded) -> None:
    before = report(loaded)
    create_views(loaded)
    assert report(loaded) == before


def test_a_schema_holding_a_table_is_not_dropped(loaded) -> None:
    with psycopg.connect(loaded) as conn:
        conn.execute("CREATE TABLE analysis.notes (note text)")
        conn.execute("INSERT INTO analysis.notes VALUES ('keep me')")
        with pytest.raises(AnalysisError, match=r"holds more than the analysis views \(notes\)"):
            install(conn.cursor())
        conn.rollback()


def test_the_views_can_not_be_read_before_they_exist(loaded) -> None:
    with psycopg.connect(loaded, autocommit=True) as conn:
        conn.execute("DROP SCHEMA analysis CASCADE")
    try:
        with pytest.raises(AnalysisError, match="cannot read the analysis views"):
            report(loaded)
    finally:
        create_views(loaded)


@pytest.mark.parametrize("days", [0, -5, True, 1.5, "30"])
def test_a_bad_ltv_window_is_refused(loaded, days) -> None:
    with pytest.raises(AnalysisError, match="ltv_days must be a whole number"):
        report(loaded, ltv_days=days)


def test_the_cli_prints_the_period_s_figures(loaded, expected, capsys) -> None:
    assert main(["--dsn", loaded, "--ltv-days", "30"]) == 0
    out = capsys.readouterr().out
    assert f"period 2025-10-01 to {expected['last_day']}" in out
    total = expected["blended"][None]
    assert f"revenue ${total['revenue']:,.2f}" in out
    for channel in ("paid_search", "paid_social", "display", "online_video"):
        assert channel in out
    assert "first 30 days" in out


def test_the_cli_needs_a_database(monkeypatch, capsys) -> None:
    monkeypatch.delenv("MARKETING_ROI_DSN", raising=False)
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "no database" in capsys.readouterr().err


def test_the_cli_refuses_a_zero_window(loaded, capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--dsn", loaded, "--ltv-days", "0"])
    assert "--ltv-days must be at least 1" in capsys.readouterr().err


def test_the_cli_reports_an_unreachable_database(capsys) -> None:
    code = main(["--dsn", "host=127.0.0.1 port=1 dbname=nothing_test connect_timeout=2"])
    assert code == 1
    assert "cannot create the analysis views" in capsys.readouterr().err
