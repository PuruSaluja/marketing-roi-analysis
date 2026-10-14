import csv
import datetime as dt
import json
import shutil
from decimal import Decimal
from pathlib import Path

import psycopg
import pytest

from marketing_roi.generate import GeneratorConfig, generate, write
from marketing_roi.load import LoadError, load, main


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("generated")
    write(generate(GeneratorConfig(seed=11, start=dt.date(2025, 11, 1), days=60)), out)
    return out


@pytest.fixture
def files(generated, tmp_path) -> Path:
    """A copy of the generated files that a test may damage."""
    return Path(shutil.copytree(generated, tmp_path / "data"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rewrite(path: Path, edit) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(edit(text), encoding="utf-8", newline="")


def schemas(db) -> list[str]:
    return [
        r[0]
        for r in db.execute(
            "SELECT nspname FROM pg_namespace WHERE nspname IN ('marketing', 'truth') ORDER BY 1"
        )
    ]


# --- a good load ----------------------------------------------------------------------------


def test_every_row_is_loaded_and_agrees_with_the_files(db, test_dsn, generated) -> None:
    counts = load(test_dsn, generated)
    sessions = rows(generated / "ga4_sessions.csv")
    channel_daily = rows(generated / "channel_daily.csv")
    truth_daily = rows(generated / "truth_daily.csv")
    sources = {(r["source"], r["medium"], r["campaign"]) for r in sessions}
    assert counts == {
        "marketing.channel_daily": len(channel_daily),
        "marketing.traffic_sources": len(sources),
        "marketing.ga4_sessions": len(sessions),
        "truth.daily": len(truth_daily),
    }
    revenue = sum(Decimal(r["purchase_revenue"]) for r in sessions if r["purchase_revenue"])
    spend = sum(Decimal(r["spend"]) for r in channel_daily)
    got = db.execute("""
        SELECT (SELECT sum(purchase_revenue) FROM marketing.ga4_sessions),
               (SELECT sum(spend) FROM marketing.channel_daily),
               (SELECT count(*) FROM marketing.ga4_sessions_flat),
               (SELECT count(transaction_id) FROM marketing.ga4_sessions)""").fetchone()
    purchases = sum(1 for r in sessions if r["transaction_id"])
    assert got == (revenue, spend, len(sessions), purchases)


def test_a_session_reads_back_as_written(db, test_dsn, generated) -> None:
    load(test_dsn, generated)
    first_purchase = next(r for r in rows(generated / "ga4_sessions.csv") if r["transaction_id"])
    got = db.execute(
        "SELECT * FROM marketing.ga4_sessions_flat WHERE session_id = %s",
        (int(first_purchase["session_id"]),),
    ).fetchone()
    (
        session_id,
        start,
        date,
        user,
        source,
        medium,
        campaign,
        group,
        device,
        engaged,
        transaction,
        revenue,
    ) = got
    assert str(session_id) == first_purchase["session_id"]
    assert start == dt.datetime.fromisoformat(first_purchase["session_start"])
    assert date == start.date()
    assert (user, source, medium, campaign, group, device, transaction) == tuple(
        first_purchase[k]
        for k in (
            "user_pseudo_id",
            "source",
            "medium",
            "campaign",
            "default_channel_group",
            "device_category",
            "transaction_id",
        )
    )
    assert engaged is True
    assert revenue == Decimal(first_purchase["purchase_revenue"])


def test_paid_sources_link_to_their_channel(db, test_dsn, generated) -> None:
    load(test_dsn, generated)
    linked = db.execute("""
        SELECT t.default_channel_group, c.channel_group
        FROM marketing.traffic_sources t LEFT JOIN marketing.channels c USING (channel_id)
        ORDER BY t.source_id""").fetchall()
    for group, channel_group in linked:
        if group in ("Direct", "Organic Search", "Referral", "Email"):
            assert channel_group is None
        else:
            assert channel_group == group


def test_truth_is_loaded_apart_from_the_marketing_tables(db, test_dsn, generated) -> None:
    load(test_dsn, generated)
    params = db.execute("SELECT params FROM truth.run").fetchone()[0]
    assert params == json.loads((generated / "truth.json").read_text(encoding="utf-8"))
    tables = db.execute("""
        SELECT table_schema, table_name FROM information_schema.tables
        WHERE table_schema IN ('marketing', 'truth') ORDER BY 1, 2""").fetchall()
    assert tables == [
        ("marketing", "channel_daily"),
        ("marketing", "channels"),
        ("marketing", "ga4_sessions"),
        ("marketing", "ga4_sessions_flat"),
        ("marketing", "traffic_sources"),
        ("truth", "daily"),
        ("truth", "run"),
    ]


# --- what the schema refuses ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("statement", "error"),
    [
        (
            "UPDATE marketing.channel_daily SET clicks = impressions + 1 "
            "WHERE (date, channel_id) = (SELECT date, channel_id FROM marketing.channel_daily "
            "LIMIT 1)",
            psycopg.errors.CheckViolation,
        ),
        (
            "UPDATE marketing.ga4_sessions SET purchase_revenue = NULL "
            "WHERE session_id = (SELECT min(session_id) FROM marketing.ga4_sessions "
            "WHERE transaction_id IS NOT NULL)",
            psycopg.errors.CheckViolation,
        ),
        (
            "UPDATE marketing.ga4_sessions SET engaged_session = false "
            "WHERE session_id = (SELECT min(session_id) FROM marketing.ga4_sessions "
            "WHERE transaction_id IS NOT NULL)",
            psycopg.errors.CheckViolation,
        ),
        (
            "UPDATE marketing.ga4_sessions SET device_category = 'watch' WHERE session_id = 1",
            psycopg.errors.CheckViolation,
        ),
        (
            "UPDATE marketing.ga4_sessions SET transaction_id = (SELECT max(transaction_id) "
            "FROM marketing.ga4_sessions), purchase_revenue = 1, engaged_session = true "
            "WHERE session_id = 1",
            psycopg.errors.UniqueViolation,
        ),
        ("INSERT INTO truth.run (params) VALUES ('{}')", psycopg.errors.UniqueViolation),
        (
            "UPDATE marketing.ga4_sessions SET source_id = 999 WHERE session_id = 1",
            psycopg.errors.ForeignKeyViolation,
        ),
    ],
)
def test_the_schema_refuses_inconsistent_rows(db, test_dsn, generated, statement, error) -> None:
    load(test_dsn, generated)
    with pytest.raises(error):
        db.execute(statement)


# --- indexes --------------------------------------------------------------------------------


QUERIES = {
    "ga4_purchases_by_date": (
        "SELECT session_date, source_id, sum(purchase_revenue) FROM marketing.ga4_sessions "
        "WHERE transaction_id IS NOT NULL AND session_date BETWEEN '2025-11-20' AND "
        "'2025-11-30' GROUP BY 1, 2"
    ),
    "ga4_purchases_by_user": (
        "SELECT DISTINCT ON (user_pseudo_id) user_pseudo_id, session_start, source_id "
        "FROM marketing.ga4_sessions WHERE transaction_id IS NOT NULL "
        "ORDER BY user_pseudo_id, session_start"
    ),
    "ga4_sessions_date_source": (
        "SELECT source_id, count(*) FROM marketing.ga4_sessions "
        "WHERE session_date = '2025-11-28' GROUP BY 1"
    ),
    "channel_daily_by_channel": (
        "SELECT date, spend FROM marketing.channel_daily WHERE channel_id = 2 ORDER BY date"
    ),
}
# Spend is not in that index, so its rows are read from the table, already in date order.
NOT_COVERING = {"channel_daily_by_channel"}


@pytest.mark.parametrize("index", sorted(QUERIES))
def test_each_index_serves_its_query(db, test_dsn, generated, index) -> None:
    load(test_dsn, generated)
    # The sample is small enough that a scan of the table is cheaper, so take that off the
    # table; what is checked is that the index fits the query and, where it covers it, that
    # the load vacuumed, so no row has to be looked up in the table.
    db.execute("SET enable_seqscan = off")
    db.execute("SET enable_bitmapscan = off")
    plan = "\n".join(
        r[0] for r in db.execute("EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) " + QUERIES[index])
    )
    assert "Sort" not in plan, plan
    if index in NOT_COVERING:
        assert f"Index Scan using {index} " in plan, plan
    else:
        assert f"Index Only Scan using {index} " in plan, plan
        assert "Heap Fetches: 0" in plan, plan


# --- refusals -------------------------------------------------------------------------------


def test_a_second_load_is_refused_and_leaves_the_first(db, test_dsn, generated) -> None:
    load(test_dsn, generated)
    before = db.execute("SELECT count(*) FROM marketing.ga4_sessions").fetchone()
    with pytest.raises(LoadError, match="already exists.*replace"):
        load(test_dsn, generated)
    assert db.execute("SELECT count(*) FROM marketing.ga4_sessions").fetchone() == before


def test_replace_loads_again(db, test_dsn, generated) -> None:
    first = load(test_dsn, generated)
    db.execute("DELETE FROM truth.daily")
    assert load(test_dsn, generated, replace=True) == first


def test_a_failed_replace_keeps_what_was_loaded(db, test_dsn, generated, files) -> None:
    load(test_dsn, generated)
    before = db.execute("SELECT count(*), sum(purchase_revenue) FROM marketing.ga4_sessions")
    before = before.fetchone()
    rewrite(files / "ga4_sessions.csv", lambda t: t[: t.index(",", len(t) * 2 // 3)])  # mid-row
    with pytest.raises(LoadError, match="refused the load"):
        load(test_dsn, files, replace=True)
    after = db.execute("SELECT count(*), sum(purchase_revenue) FROM marketing.ga4_sessions")
    assert after.fetchone() == before


def test_a_failed_first_load_leaves_no_schema(db, test_dsn, files) -> None:
    rewrite(files / "channel_daily.csv", lambda t: t.replace(",paid_social,", ",paid_socal,", 1))
    with pytest.raises(LoadError, match="paid_socal"):
        load(test_dsn, files)
    assert schemas(db) == []


def test_a_renamed_column_is_refused(db, test_dsn, files) -> None:
    rewrite(files / "ga4_sessions.csv", lambda t: t.replace("user_pseudo_id", "user_id", 1))
    with pytest.raises(LoadError, match="user_id"):
        load(test_dsn, files)
    assert schemas(db) == []


def test_a_missing_file_is_named(db, test_dsn, files) -> None:
    (files / "truth_daily.csv").unlink()
    with pytest.raises(LoadError, match="lacks truth_daily.csv"):
        load(test_dsn, files)
    assert schemas(db) == []


def test_rows_missing_against_truth_json_are_refused(db, test_dsn, files) -> None:
    rewrite(files / "ga4_sessions.csv", lambda t: t.rstrip("\r\n").rsplit("\n", 1)[0] + "\n")
    with pytest.raises(LoadError, match="ga4_sessions has .* truth.json says"):
        load(test_dsn, files)
    assert schemas(db) == []


def test_revenue_that_disagrees_with_truth_is_refused(db, test_dsn, files) -> None:
    sessions = rows(files / "ga4_sessions.csv")
    bought = next(r for r in sessions if r["transaction_id"])
    bought["purchase_revenue"] = str(Decimal(bought["purchase_revenue"]) + Decimal("0.01"))
    with (files / "ga4_sessions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sessions[0]))
        w.writeheader()
        w.writerows(sessions)
    with pytest.raises(LoadError, match="session revenue .* differs"):
        load(test_dsn, files)
    assert schemas(db) == []


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "not valid JSON"),
        ("[]", "no 'channels' and 'rows'"),
        (
            '{"channels": {"paid_search": {"source": "google"}}, "rows": {}}',
            "'paid_search' lacks medium, channel_group, flighted",
        ),
    ],
)
def test_a_damaged_truth_json_is_refused(db, test_dsn, files, content, message) -> None:
    (files / "truth.json").write_text(content, encoding="utf-8")
    with pytest.raises(LoadError, match=message):
        load(test_dsn, files)
    assert schemas(db) == []


def test_an_unreachable_database_is_a_load_error(files) -> None:
    with pytest.raises(LoadError, match="refused the load"):
        load("host=127.0.0.1 port=1 dbname=nowhere_test connect_timeout=2", files)


# --- the command line -----------------------------------------------------------------------


def test_cli_loads_and_reports(db, test_dsn, generated, capsys) -> None:
    assert main(["--data", str(generated), "--dsn", test_dsn]) == 0
    out = capsys.readouterr().out
    assert "marketing.ga4_sessions" in out and "loaded in" in out


def test_cli_reports_a_refusal(db, test_dsn, generated, capsys) -> None:
    assert main(["--data", str(generated), "--dsn", test_dsn]) == 0
    assert main(["--data", str(generated), "--dsn", test_dsn]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["--data", str(generated), "--dsn", test_dsn, "--replace"]) == 0


def test_cli_needs_a_database(monkeypatch, capsys) -> None:
    monkeypatch.delenv("MARKETING_ROI_DSN", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--data", "anywhere"])
    assert exc.value.code == 2
    assert "MARKETING_ROI_DSN" in capsys.readouterr().err
