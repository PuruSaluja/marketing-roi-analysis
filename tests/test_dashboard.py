"""The Plotly dashboard: drawn from the extracts alone, its figures are the analysis
views' figures, and its charts draw them."""

import datetime as dt
import math

import psycopg
import pytest
from psycopg.rows import dict_row
from test_extracts import make_run, save_runs

from marketing_roi import dashboard, extracts
from marketing_roi.analysis import create_views, report
from marketing_roi.generate import GeneratorConfig, generate, write
from marketing_roi.load import load

# Crosses New Year and ends on 2026-01-30, so the last month is incomplete.
CONFIG = GeneratorConfig(seed=5, start=dt.date(2025, 11, 1), days=91)


@pytest.fixture(scope="module")
def loaded(test_dsn, tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("generated")
    write(generate(CONFIG), out)
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS analysis, marketing, truth CASCADE")
    load(test_dsn, out)
    create_views(test_dsn)
    return test_dsn


@pytest.fixture(scope="module")
def folder(loaded, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("extracts")
    save_runs(tmp / "optimise.json", make_run(loaded))
    extracts.extract(loaded, tmp / "out", optimiser=tmp / "optimise.json", ltv_days=30)
    return tmp / "out"


@pytest.fixture(scope="module")
def s(folder):
    return dashboard.summary(dashboard.read(folder))


def view(dsn: str, sql: str, *params) -> list[dict]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        return conn.execute(sql, params).fetchall()


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def test_channel_figures_are_the_view_s(loaded, s) -> None:
    rows = view(loaded, "SELECT * FROM analysis.channel_performance WHERE month IS NULL "
                "ORDER BY spend DESC, channel")  # fmt: skip
    assert [c["channel"] for c in s["channels"]] == [r["channel"] for r in rows]
    for c, r in zip(s["channels"], rows, strict=True):
        for k in ("spend", "impressions", "clicks", "sessions", "purchases", "revenue", "ctr",
                  "cpc", "conversion_rate", "cost_per_purchase", "roas"):  # fmt: skip
            assert same(c[k], r[k]), (c["channel"], k, c[k], r[k])


def test_monthly_figures_are_the_view_s(loaded, s) -> None:
    rows = view(loaded, "SELECT * FROM analysis.channel_performance WHERE month IS NOT NULL")
    assert len(s["monthly"]) == len(rows) == 4 * 3
    for r in rows:
        cell = s["monthly"][(r["month"], r["channel"])]
        assert same(cell["spend"], r["spend"]) and same(cell["revenue"], r["revenue"])


def test_acquisition_and_blended_figures_are_the_views(loaded, s) -> None:
    rows = view(loaded, "SELECT * FROM analysis.acquisition WHERE month IS NULL")
    by_group = {g["channel_group"]: g for g in s["acquisition"]}
    assert set(by_group) == {r["channel_group"] for r in rows}
    for r in rows:
        g = by_group[r["channel_group"]]
        assert g["paid"] == r["paid"]
        for k in ("spend", "new_customers", "cac"):
            assert same(g[k], r[k]), (r["channel_group"], k)
    (b,) = view(loaded, "SELECT * FROM analysis.blended WHERE month IS NULL")
    for k in ("spend", "revenue", "new_customers", "new_customer_revenue", "paid_revenue",
              "blended_roas", "blended_roi", "paid_roas", "blended_cac"):  # fmt: skip
        assert same(s["kpis"][k], b[k]), k
    assert (s["kpis"]["first_day"], s["kpis"]["last_day"]) == (
        CONFIG.start,
        CONFIG.start + dt.timedelta(days=90),
    )


def test_retention_is_the_report_s_and_the_view_s(loaded, s) -> None:
    pooled = report(loaded, ltv_days=30)["retention"]
    assert pooled  # the November cohort's December
    assert list(s["retention"]) == [r["months_since"] for r in pooled]
    for r in pooled:
        assert same(s["retention"][r["months_since"]], r["retention"])
    complete = view(loaded, "SELECT * FROM analysis.cohorts WHERE complete")
    assert len(s["cohort_matrix"]) == len(complete) == 3  # Nov 0-1, Dec 0
    for r in complete:
        assert same(s["cohort_matrix"][(r["cohort_month"], r["months_since"])], r["retention"])


def by_title(figs, start: str):
    (fig,) = [f for t, f in figs if t.startswith(start)]
    return fig


def test_the_charts_draw_the_figures(s) -> None:
    figs = dashboard.figures(s)
    assert len(figs) == 8
    perf = by_title(figs, "Channel performance")
    assert list(perf.data[0].y) == [c["roas"] for c in s["channels"]]
    assert list(perf.data[1].y) == [c["cost_per_purchase"] for c in s["channels"]]
    assert list(perf.data[0].x) == [c["channel_group"] for c in s["channels"]]

    monthly = by_title(figs, "Spend and last-click revenue")
    first = s["channels"][0]["channel"]
    months = sorted({m for m, _ in s["monthly"]})
    assert list(monthly.data[0].y) == [s["monthly"][(m, first)]["spend"] for m in months]
    assert list(monthly.data[4].y) == [s["monthly"][(m, first)]["revenue"] for m in months]
    # A channel's revenue line is drawn in its spend bars' colour.
    assert monthly.data[0].marker.color == monthly.data[4].line.color

    acq = by_title(figs, "Customer acquisition")
    paid = [g for g in s["acquisition"] if g["paid"]]
    assert list(acq.data[1].y) == [g["cac"] for g in paid]

    heat = by_title(figs, "Cohort retention")
    nov = s["cohort_matrix"][(dt.date(2025, 11, 1), 1)]
    assert heat.data[0].z[0][0] == nov and heat.data[0].y[0] == "2025-11"

    ltv = by_title(figs, "Customer value against CAC (first 30 days)")
    assert list(ltv.data[1].y) == [r["cac"] for r in s["ltv"] if r["paid"]]


def test_the_budget_charts_label_projected_and_true(s) -> None:
    figs = dashboard.figures(s)
    budget = by_title(figs, "Daily budget")
    assert [t.name for t in budget.data] == ["current", "recommendation", "true optimum"]
    order = [c["channel"] for c in s["channels"]]
    want = {(r["split"], r["channel"]): r["spend_per_day"] for r in s["optimiser"]["splits"]}
    assert list(budget.data[1].y) == [want[("recommendation", c)] for c in order]

    gains = by_title(figs, "Gain over the current budgets")
    assert list(gains.data[0].x) == ["committed", "prior sd 0.5", "recommendation",
                                     "true optimum"]  # fmt: skip
    assert list(gains.data[0].y) == [0.05, 0.03, 0.025, None]
    assert list(gains.data[1].y) == [0.012, 0.012, 0.012, 0.015]
    assert "projected" in gains.data[0].name and "true" in gains.data[1].name

    hold = by_title(figs, "Held-out quarter from 2025-12-31")
    assert list(hold.data[0].x) == ["committed", "prior sd 0.5", "recommendation"]
    assert list(hold.data[0].y) == [900.0, 300.0, 300.0]
    assert list(hold.data[0].error_y.array) == [0.0, 600.0, 600.0]  # up to the most, 900
    assert list(hold.data[0].error_y.arrayminus) == [600.0, 0.0, 0.0]  # down to the least
    assert list(hold.data[1].y) == [100.0, 100.0, 100.0, 150.0]


def test_the_page_carries_the_notes_and_the_figures(folder, tmp_path, s) -> None:
    path = tmp_path / "page" / "dashboard.html"
    size = dashboard.write(s, path)
    text = path.read_text(encoding="utf-8")
    assert size == len(text.encode("utf-8")) > 3_000_000  # Plotly's JavaScript is in it
    assert 'src="https://cdn.plot.ly' not in text
    for note in ("The data is synthetic", "last click", "Projected is not measured",
                 f"${s['kpis']['spend']:,.0f}", f"{s['kpis']['blended_roi']:.1%}"):  # fmt: skip
        assert note in text
    assert text.count("<h2>") == 8

    small = tmp_path / "cdn.html"
    assert dashboard.main(["--extracts", str(folder), "--out", str(small), "--cdn"]) == 0
    linked = small.read_text(encoding="utf-8")
    assert linked.count('src="https://cdn.plot.ly/') == 1 and len(linked) < 1_000_000


def test_without_optimiser_files_the_budget_charts_are_left_out(loaded, tmp_path) -> None:
    extracts.extract(loaded, tmp_path / "views")  # 180 days: no customer followed that long
    s = dashboard.summary(dashboard.read(tmp_path / "views"))
    assert s["optimiser"] is None
    titles = [t for t, _ in dashboard.figures(s)]
    assert titles == ["Spend and last-click revenue by month",
                      "Channel performance, by last click",
                      "Customer acquisition, by the group credited with the first purchase",
                      "Cohort retention", "Customer value against CAC"]  # fmt: skip
    text = dashboard.page(s, cdn=True)
    assert "No paid customer is followed for the whole window" in text
    assert "budget charts are left out" in text and "Projected is not measured" not in text


def test_some_optimiser_files_without_the_others_are_refused(folder, tmp_path) -> None:
    for name in (*extracts.VIEW_FILES, "budget_gains"):
        (tmp_path / f"{name}.csv").write_bytes((folder / f"{name}.csv").read_bytes())
    with pytest.raises(dashboard.DashboardError, match="not budget_splits, holdout_lift"):
        dashboard.read(tmp_path)


def copy_views(folder, tmp_path):
    for name in extracts.VIEW_FILES:
        (tmp_path / f"{name}.csv").write_bytes((folder / f"{name}.csv").read_bytes())
    return tmp_path


def edit_first_row(path, change) -> None:
    """Rewrite the first row's fields (the file's line 2) with ``change``."""
    lines = path.read_text(encoding="utf-8").split("\n")
    lines[1] = ",".join(change(lines[1].split(",")))
    path.write_text("\n".join(lines), encoding="utf-8")


def header_only(path) -> None:
    path.write_text(path.read_text(encoding="utf-8").split("\n")[0] + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda p: p.unlink(), r"has no channel_daily\.csv"),
        (lambda p: p.write_text("date,spend\n2025-01-01,1\n", encoding="utf-8"), "has columns"),
        (lambda p: edit_first_row(p, lambda f: [*f[:4], "x" + f[4], *f[5:]]),
         "line 2: could not convert"),
        (lambda p: edit_first_row(p, lambda f: f[:-1]), "line 2 has 9 fields"),
        (lambda p: p.write_bytes(b"date,\xff\n"), "not UTF-8"),
        (header_only, "has no rows"),
    ],
)  # fmt: skip
def test_a_damaged_extract_is_refused_plainly(folder, tmp_path, capsys, damage, message) -> None:
    copy_views(folder, tmp_path)
    damage(tmp_path / "channel_daily.csv")
    with pytest.raises(dashboard.DashboardError, match=message):
        dashboard.read(tmp_path)
    assert dashboard.main(["--extracts", str(tmp_path), "--out", str(tmp_path / "d.html")]) == 1
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "d.html").exists()


@pytest.mark.parametrize("text, message", [("2025-01-01,nan", "not a finite number"),
                                           ("2025-01-01,maybe", "not true or false")])  # fmt: skip
def test_a_field_of_the_wrong_kind_is_refused(tmp_path, text, message) -> None:
    columns = {"day": "date", "x": "bool" if "maybe" in text else "number"}
    path = tmp_path / "t.csv"
    path.write_text("day,x\n" + text + "\n", encoding="utf-8")
    dashboard.COLUMNS["_t"] = columns
    try:
        with pytest.raises(dashboard.DashboardError, match=f"line 2: {message}"):
            dashboard.read_csv(path, "_t")
    finally:
        del dashboard.COLUMNS["_t"]


def test_a_folder_that_is_not_there_is_refused(tmp_path) -> None:
    with pytest.raises(dashboard.DashboardError, match="not a folder of extracts"):
        dashboard.read(tmp_path / "missing")


def test_an_out_path_that_is_a_folder_is_refused(folder, tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        dashboard.main(["--extracts", str(folder), "--out", str(tmp_path)])
    assert "is a folder" in capsys.readouterr().err


def test_a_page_that_cannot_be_written_is_refused(s, tmp_path) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(dashboard.DashboardError, match="cannot write"):
        dashboard.write(s, tmp_path / "file" / "d.html", cdn=True)  # a file as its folder
