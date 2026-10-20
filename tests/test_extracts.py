"""The CSV extracts: the views' figures, re-added from the files, come back as the views
give them, and an optimiser run is taken only if it is of the loaded data."""

import csv
import datetime as dt
import json
from decimal import Decimal

import psycopg
import pytest
from psycopg.rows import dict_row

from marketing_roi import extracts
from marketing_roi.analysis import create_views
from marketing_roi.generate import GeneratorConfig, generate, write
from marketing_roi.load import load

# Crosses New Year and ends on 2026-01-30, so the last month is incomplete.
CONFIG = GeneratorConfig(seed=5, start=dt.date(2025, 11, 1), days=91)
TRAIN_DAYS = 60
CHANNELS = ("paid_search", "paid_social", "display", "online_video")


@pytest.fixture(scope="module")
def loaded(test_dsn, tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("generated")
    write(generate(CONFIG), out)
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS analysis, marketing, truth CASCADE")
    load(test_dsn, out)
    create_views(test_dsn)
    return test_dsn


def view(dsn: str, sql: str, *params) -> list[dict]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        return conn.execute(sql, params).fetchall()


def current_budgets(dsn: str, days: int = TRAIN_DAYS) -> dict[str, float]:
    """Each channel's mean daily spend over the first ``days``, as the optimiser takes it."""
    rows = view(
        dsn,
        "SELECT channel, sum(spend) AS spend FROM analysis.channel_daily "
        "WHERE date < (SELECT min(date) FROM analysis.channel_daily) + %s GROUP BY 1",
        days,
    )
    return {r["channel"]: float(r["spend"]) / days for r in rows}


def make_run(dsn: str, seed: int = 5) -> dict:
    """An optimiser run in the shape ``optimise --json`` writes, for the loaded data."""
    current = current_budgets(dsn)
    moved = dict(current, paid_search=current["paid_search"] - 10.0,
                 paid_social=current["paid_social"] + 10.0)  # fmt: skip
    best = dict(current, paid_search=current["paid_search"] - 20.0,
                paid_social=current["paid_social"] + 20.0)  # fmt: skip
    fit = dict(spend=moved, projected_gain=30.0, projected_gain_share=0.03, true_gain=12.0,
               true_gain_share=0.012, holdout_mape=0.15)  # fmt: skip
    projected = {"committed": 900.0, "prior_0.5": 300.0}
    return {
        "seed": seed,
        "truth": "default",
        "train_days": TRAIN_DAYS,
        "current": current,
        "new_customer_revenue": 1000.0,
        "fits": {"committed": dict(fit, projected_gain=50.0, projected_gain_share=0.05),
                 "prior_0.5": fit},
        "robust": {"from": "prior_0.5", "worst_projected_gain": 25.0,
                   "worst_projected_share": 0.025},
        "recommendation": {"change": True, "from": "prior_0.5", "spend": moved,
                           "true_gain": 12.0, "true_gain_share": 0.012},
        "true_optimum": {"spend": best, "true_gain": 15.0, "true_gain_share": 0.015},
        "holdout": {
            "days": 31, "first_day": "2025-12-31", "spend": 5000.0,
            "new_customer_revenue": 9000.0,
            "splits": {
                "committed": {"projected": projected, "true": 100.0},
                "prior_0.5": {"projected": projected, "true": 100.0},
                "recommendation": {"projected": projected, "true": 100.0},
                "true_optimum": {"projected": {"committed": 400.0, "prior_0.5": 700.0},
                                 "true": 150.0},
            },
        },
    }  # fmt: skip


def save_runs(path, *runs) -> None:
    path.write_text(json.dumps({"runs": list(runs)}), encoding="utf-8")


def read_back(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def written(loaded, tmp_path_factory):
    folder = tmp_path_factory.mktemp("extracts")
    results = folder / "optimise.json"
    save_runs(results, make_run(loaded))
    rows = extracts.extract(loaded, folder / "out", optimiser=results, ltv_days=30)
    return folder / "out", rows


def test_every_file_is_written_with_its_columns(written) -> None:
    out, rows = written
    assert set(rows) == set(extracts.COLUMNS)
    for name, columns in extracts.COLUMNS.items():
        with open(out / f"{name}.csv", newline="", encoding="utf-8") as fh:
            assert next(csv.reader(fh)) == list(columns)
        assert len(read_back(out / f"{name}.csv")) == rows[name]
    assert rows["channel_daily"] == 4 * 91


def test_daily_rows_add_up_to_the_channel_view(loaded, written) -> None:
    out, _ = written
    totals: dict[str, dict] = {}
    for r in read_back(out / "channel_daily.csv"):
        t = totals.setdefault(r["channel"], dict.fromkeys(extracts.COLUMNS["channel_daily"], 0))
        for k in ("spend", "impressions", "clicks", "sessions", "purchases", "revenue"):
            t[k] += Decimal(r[k])
        assert r["month"] == r["date"][:8] + "01"
    rows = view(loaded, "SELECT * FROM analysis.channel_performance WHERE month IS NULL")
    assert len(rows) == len(totals) == 4
    for r in rows:
        for k in ("spend", "impressions", "clicks", "sessions", "purchases", "revenue"):
            assert totals[r["channel"]][k] == r[k], (r["channel"], k)


def test_monthly_rows_add_up_to_the_whole_period(loaded, written) -> None:
    out, _ = written
    acquired: dict[str, list] = {}
    for r in read_back(out / "acquisition_monthly.csv"):
        a = acquired.setdefault(r["channel_group"], [Decimal(0), 0, r["paid"]])
        a[0] += Decimal(r["spend"])
        a[1] += int(r["new_customers"])
    whole = view(loaded, "SELECT * FROM analysis.acquisition WHERE month IS NULL")
    assert {r["channel_group"]: [r["spend"], r["new_customers"], str(r["paid"]).lower()]
            for r in whole} == acquired  # fmt: skip

    blended = {k: Decimal(0) for k in extracts.COLUMNS["blended_monthly"] if k != "month"}
    monthly = read_back(out / "blended_monthly.csv")
    for r in monthly:
        for k in blended:
            blended[k] += Decimal(r[k])
    (period,) = view(loaded, "SELECT * FROM analysis.blended WHERE month IS NULL")
    assert blended == {k: Decimal(period[k]) for k in blended}
    assert [r["month"] for r in monthly] == ["2025-11-01", "2025-12-01", "2026-01-01"]


def test_cohort_totals_run_on_as_the_view_s_per_customer_figures(loaded, written) -> None:
    out, _ = written
    rows = read_back(out / "cohorts.csv")
    by_key = {(r["cohort_month"], int(r["months_since"])): r for r in rows}
    cohorts = view(loaded, "SELECT * FROM analysis.cohorts")
    assert len(cohorts) == len(rows) == 6  # 3 + 2 + 1 months
    for c in cohorts:
        r = by_key[(c["cohort_month"].isoformat(), c["months_since"])]
        # PostgreSQL rounds a numeric quotient to its own scale, so compare as floats.
        size = float(r["cohort_size"])
        for total, per_customer in [("cumulative_revenue", "cumulative_revenue_per_customer"),
                                    ("cumulative_purchases", "cumulative_purchases_per_customer"),
                                    ("active_customers", "retention")]:  # fmt: skip
            assert float(r[total]) / size == pytest.approx(float(c[per_customer]), rel=1e-12)
        assert r["complete"] == str(c["complete"]).lower()
    assert {r["complete"] for r in rows} == {"true", "false"}


def test_ltv_rows_are_the_function_s_at_the_window(loaded, written) -> None:
    out, _ = written
    rows = read_back(out / "ltv_by_channel.csv")
    want = view(loaded, "SELECT * FROM analysis.ltv_by_channel(30)")
    assert len(rows) == len(want) >= 4
    for r, w in zip(rows, want, strict=True):
        assert r["window_days"] == "30"
        assert r["channel_group"] == w["channel_group"]
        for k in ("customers", "spend", "cac", "revenue_per_customer", "ltv_to_cac"):
            assert (r[k] == "") == (w[k] is None)
            if w[k] is not None:
                assert Decimal(r[k]) == w[k]


def test_the_optimiser_files_hold_every_split(written) -> None:
    out, _ = written
    splits = read_back(out / "budget_splits.csv")
    names = ["current", "committed", "prior_0.5", "recommendation", "true_optimum"]
    assert [r["split"] for r in splits] == [n for n in names for _ in CHANNELS]
    assert {r["kind"] for r in splits} == {"current", "fit", "recommendation", "true optimum"}
    search = {r["split"]: float(r["spend_per_day"]) for r in splits
              if r["channel"] == "paid_search"}  # fmt: skip
    current = float(splits[0]["current_per_day"])
    assert search["current"] == current
    assert search["recommendation"] == pytest.approx(current - 10)
    assert search["true_optimum"] == pytest.approx(current - 20)
    assert {r["channel_group"] for r in splits} == {"Paid Search", "Paid Social", "Display",
                                                    "Paid Video"}  # fmt: skip

    gains = {r["split"]: r for r in read_back(out / "budget_gains.csv")}
    assert list(gains) == names
    assert gains["committed"]["projected_gain_per_day"] == "50.0"
    # The recommendation's projection is the least any fit makes for it.
    assert gains["recommendation"]["projected_gain_per_day"] == "25.0"
    assert gains["recommendation"]["projected_gain_share"] == "0.025"
    assert gains["true_optimum"]["projected_gain_per_day"] == ""
    assert gains["true_optimum"]["true_gain_per_day"] == "15.0"

    lifts = {r["split"]: r for r in read_back(out / "holdout_lift.csv")}
    assert lifts["committed"]["own_projected_lift"] == "900.0"
    assert lifts["prior_0.5"]["own_projected_lift"] == "300.0"
    assert lifts["recommendation"]["own_projected_lift"] == "300.0"
    assert lifts["true_optimum"]["own_projected_lift"] == ""
    assert lifts["true_optimum"]["least_projected_lift"] == "400.0"
    assert lifts["true_optimum"]["most_projected_lift"] == "700.0"
    assert lifts["committed"]["first_day"] == "2025-12-31"


def test_a_recommendation_to_keep_the_budget_projects_no_gain(loaded, tmp_path) -> None:
    run = make_run(loaded)
    run["recommendation"] = dict(run["recommendation"], change=False, spend=run["current"],
                                 true_gain=0.0, true_gain_share=0.0)  # fmt: skip
    save_runs(tmp_path / "keep.json", run)
    extracts.extract(loaded, tmp_path / "out", optimiser=tmp_path / "keep.json")
    gains = {r["split"]: r for r in read_back(tmp_path / "out" / "budget_gains.csv")}
    assert gains["recommendation"]["projected_gain_per_day"] == "0.0"


def test_the_views_alone_are_written_without_an_optimiser_file(loaded, tmp_path) -> None:
    rows = extracts.extract(loaded, tmp_path / "views")
    assert set(rows) == set(extracts.VIEW_FILES)
    assert sorted(p.name for p in (tmp_path / "views").iterdir()) == sorted(
        f"{n}.csv" for n in extracts.VIEW_FILES
    )
    # The default 180 days: the run's 91 follow no customer for that long.
    assert read_back(tmp_path / "views" / "ltv_by_channel.csv") == []
    assert view(loaded, "SELECT * FROM analysis.ltv_by_channel(180)") == []


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda run: run["current"].update(paid_search=run["current"]["paid_search"] * 1.01),
         "not of the loaded data"),
        (lambda run: run.update(train_days=TRAIN_DAYS + 1), "not of the loaded data"),
        (lambda run: run["current"].pop("display"), "channels"),
        (lambda run: run["current"].update(tv=1.0), "channels"),
        # A held-out quarter that is not the data's last days.
        (lambda run: run["holdout"].update(days=30), r"covers 60 \+ 30 days, the loaded data 91"),
        (lambda run: run.update(train_days=59) or run["holdout"].update(days=32),
         "not of the loaded data"),
        (lambda run: run["holdout"].update(first_day="2025-12-01"),
         "starts on 2025-12-01, the loaded data's last 31 days on 2025-12-31"),
        # The other truths keep the default's spend, so only the truth tells them apart.
        (lambda run: run.update(truth="moved"), "'moved' truth"),
        (lambda run: run.update(truth="low"), "'low' truth"),
    ],
)  # fmt: skip
def test_a_run_of_other_data_is_refused_and_nothing_is_written(
    loaded, tmp_path, change, message
) -> None:
    run = make_run(loaded)
    change(run)
    save_runs(tmp_path / "other.json", run)
    with pytest.raises(extracts.ExtractError, match=message):
        extracts.extract(loaded, tmp_path / "out", optimiser=tmp_path / "other.json")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda run: run.pop("robust"), r"missing part of a run: KeyError\('robust'\)"),
        (lambda run: run.pop("holdout"), "no held-out quarter"),
        (lambda run: run.update(holdout="2025-12-31"), "no held-out quarter"),
        (lambda run: run["current"].update(display=float("nan")),
         "current budgets are not numbers"),
        (lambda run: run.pop("current"), "current budgets are not numbers"),
        (lambda run: run["current"].update(display="abc"), "current budgets are not numbers"),
        (lambda run: run["current"].update(display=None), "current budgets are not numbers"),
        (lambda run: run.update(train_days=0), "training period is not a whole number"),
        (lambda run: run.update(train_days=60.5), "training period is not a whole number"),
        (lambda run: run.update(train_days="60"), "training period is not a whole number"),
        (lambda run: run.update(train_days=True), "training period is not a whole number"),
        (lambda run: run["holdout"].update(days=0), "held-out quarter is not a whole number"),
        (lambda run: run["holdout"].update(first_day="2025-13-01"), "has no first day"),
        (lambda run: run["holdout"].update(first_day=None), "has no first day"),
        (lambda run: run["recommendation"].update(change="no"), "not true or false"),
        (lambda run: run["holdout"]["splits"]["committed"].update(projected={}),
         "holdout.splits.committed.projected"),
    ],
)  # fmt: skip
def test_a_run_missing_or_damaging_part_of_its_results_is_refused_plainly(
    loaded, tmp_path, change, message
) -> None:
    run = make_run(loaded)
    change(run)
    save_runs(tmp_path / "part.json", run)
    with pytest.raises(extracts.ExtractError, match=message):
        extracts.extract(loaded, tmp_path / "out", optimiser=tmp_path / "part.json")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("runs", [[1], [1, 2], ["x"]])
def test_runs_that_are_not_records_are_refused_plainly(tmp_path, runs) -> None:
    (tmp_path / "r.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    with pytest.raises(extracts.ExtractError, match="a run is not a record"):
        extracts.read_optimiser(tmp_path / "r.json", seed=5 if len(runs) > 1 else None)


def test_a_run_written_before_the_truth_was_recorded_is_taken_as_the_default(
    loaded, tmp_path
) -> None:
    run = make_run(loaded)
    del run["truth"]
    save_runs(tmp_path / "old.json", run)
    assert extracts.extract(loaded, tmp_path / "out", optimiser=tmp_path / "old.json")


def test_the_recommendation_s_share_is_the_run_s_own(loaded, tmp_path) -> None:
    run = make_run(loaded)
    run["new_customer_revenue"] = 0.0  # the share is the run's, not divided again here
    save_runs(tmp_path / "share.json", run)
    extracts.extract(loaded, tmp_path / "out", optimiser=tmp_path / "share.json")
    gains = {r["split"]: r for r in read_back(tmp_path / "out" / "budget_gains.csv")}
    assert gains["recommendation"]["projected_gain_share"] == "0.025"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_figure_a_csv_cannot_hold_is_refused_before_any_file(loaded, tmp_path, bad) -> None:
    run = make_run(loaded)
    run["fits"]["committed"]["projected_gain"] = bad
    save_runs(tmp_path / "bad.json", run)
    with pytest.raises(extracts.ExtractError, match="cannot hold"):
        extracts.extract(loaded, tmp_path / "out", optimiser=tmp_path / "bad.json")
    assert not (tmp_path / "out").exists()


def test_a_run_is_picked_by_seed_from_a_file_of_several(loaded, tmp_path) -> None:
    other = make_run(loaded, seed=6)
    other["current"] = {c: 1.0 for c in CHANNELS}  # not this data
    save_runs(tmp_path / "many.json", other, make_run(loaded))
    with pytest.raises(extracts.ExtractError, match=r"2 runs \(seeds 6, 5\): pick one"):
        extracts.read_optimiser(tmp_path / "many.json")
    assert extracts.read_optimiser(tmp_path / "many.json", 5)["seed"] == 5
    with pytest.raises(extracts.ExtractError, match="no run with seed 7"):
        extracts.read_optimiser(tmp_path / "many.json", 7)
    many = ["--dsn", loaded, "--optimise", str(tmp_path / "many.json")]
    assert extracts.main([*many, "--out", str(tmp_path / "out"), "--seed", "5"]) == 0
    assert extracts.main([*many, "--out", str(tmp_path / "out6"), "--seed", "6"]) == 1


@pytest.mark.parametrize(
    "content, message",
    [(None, "cannot read"), ("{not json", "not an optimiser results file"),
     ('{"other": 1}', "not an optimiser results file"), ('{"runs": []}', "holds no runs"),
     ("[1, 2]", "not an optimiser results file")],
)  # fmt: skip
def test_a_file_that_is_not_optimiser_results_is_refused_plainly(
    tmp_path, capsys, content, message
) -> None:
    path = tmp_path / "results.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(extracts.ExtractError, match=message):
        extracts.read_optimiser(path)
    # Read before the database, so no connection is needed to refuse it.
    assert extracts.main(["--dsn", "dbname=unused", "--out", str(tmp_path / "out"),
                          "--optimise", str(path)]) == 1  # fmt: skip
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_an_out_path_that_is_a_file_is_refused_before_the_database(tmp_path, capsys) -> None:
    path = tmp_path / "taken.csv"
    path.write_text("x", encoding="utf-8")
    assert extracts.main(["--dsn", "dbname=unused", "--out", str(path)]) == 1
    assert "is a file, not a folder" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == "x"


def test_the_command_line_writes_the_files(loaded, tmp_path, capsys) -> None:
    assert extracts.main(["--dsn", loaded, "--out", str(tmp_path / "x"), "--ltv-days", "60"]) == 0
    out = capsys.readouterr().out
    assert "channel_daily.csv: 364 rows" in out and "wrote 5 files" in out
    assert read_back(tmp_path / "x" / "ltv_by_channel.csv")[0]["window_days"] == "60"


@pytest.mark.parametrize(
    "argv, message",
    [([], "no database"), (["--dsn", "x", "--ltv-days", "0"], "at least 1"),
     (["--dsn", "x", "--seed", "5"], "--seed picks a run")],
)  # fmt: skip
def test_the_command_line_refuses_bad_arguments(monkeypatch, capsys, argv, message) -> None:
    monkeypatch.delenv("MARKETING_ROI_DSN", raising=False)
    with pytest.raises(SystemExit):
        extracts.main([*argv, "--out", "x"])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_bad_window_is_refused(bad) -> None:
    with pytest.raises(extracts.ExtractError, match="ltv_days"):
        extracts.read_views("dbname=unused", ltv_days=bad)


def test_a_database_without_the_views_is_refused(test_dsn) -> None:
    # A database that does not exist, so the loaded one the other tests read stays.
    with pytest.raises(extracts.ExtractError, match="cannot read the analysis views"):
        extracts.read_views(f"{test_dsn} dbname=marketing_roi_no_such_test")


def test_fields_are_written_as_tableau_reads_them() -> None:
    f = extracts._field
    assert [f(None), f(True), f(False), f(dt.date(2025, 1, 2)), f(Decimal("1.50")),
            f(0.1), f(7)] == ["", "true", "false", "2025-01-02", "1.50", "0.1", "7"]  # fmt: skip
    assert f(Decimal("1E+3")) == "1000"
