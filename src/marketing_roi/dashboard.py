"""The dashboard in ``docs/dashboard-spec.md``, drawn with Plotly from the CSV extracts.

``python -m marketing_roi.dashboard --extracts extracts --out dashboard.html`` reads only
the files ``marketing_roi.extracts`` writes, as a Tableau workbook on them would, and
writes one HTML page: the headline figures, then a chart per sheet of the spec. Every
ratio is taken over sums, as the spec's calculated fields take it, so the page's figures
are the analysis views' figures; the tests check that they are.

By default the page carries Plotly's JavaScript (about 4.7 MB) so it opens offline;
``--cdn`` links it from Plotly's CDN instead.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import math
import sys
from collections import defaultdict
from pathlib import Path

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marketing_roi.extracts import COLUMNS, OPTIMISER_FILES, VIEW_FILES

COHORT_MONTHS = 12
"""Months after the first purchase that the retention heatmap and curve show."""
PALETTE = ("#4c78a8", "#f58518", "#54a24b", "#b279a2", "#72b7b2", "#9d755d")
"""Channel colours, by the channels' order of spend, so a channel keeps its colour."""
PROJECTED, TRUE, CURRENT = "#4c78a8", "#e45756", "#9da5ad"


class DashboardError(Exception):
    """The extracts cannot be read, or the page cannot be written."""


def _parse(kind: str, text: str):
    if text == "":
        return None
    if kind == "date":
        return dt.date.fromisoformat(text)
    if kind == "bool":
        if text not in ("true", "false"):
            raise ValueError(f"not true or false: {text!r}")
        return text == "true"
    if kind == "number":
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"not a finite number: {text!r}")
        return value
    return text


def read_csv(path: Path, name: str) -> list[dict]:
    """One extract, with each column read as the type ``COLUMNS`` gives it."""
    types = COLUMNS[name]
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header != list(types):
                raise DashboardError(
                    f"{path} has columns {header}, expected {list(types)}: "
                    "write it again with marketing_roi.extracts"
                )
            rows = []
            for line, values in enumerate(reader, 2):
                if len(values) != len(header):
                    raise DashboardError(f"{path} line {line} has {len(values)} fields")
                try:
                    pairs = zip(header, values, strict=True)
                    rows.append({c: _parse(types[c], v) for c, v in pairs})
                except ValueError as exc:
                    raise DashboardError(f"{path} line {line}: {exc}") from exc
    except OSError as exc:
        raise DashboardError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise DashboardError(f"{path} is not UTF-8 text: {exc}") from exc
    return rows


def read(folder: Path) -> dict[str, list[dict]]:
    """Every extract in ``folder``. The view files are required; the optimiser's are
    read if all three are there."""
    folder = Path(folder)
    if not folder.is_dir():
        raise DashboardError(f"{folder} is not a folder of extracts")
    missing = [n for n in VIEW_FILES if not (folder / f"{n}.csv").is_file()]
    if missing:
        raise DashboardError(
            f"{folder} has no {', '.join(f'{n}.csv' for n in missing)}: "
            "write the extracts with marketing_roi.extracts first"
        )
    present = [n for n in OPTIMISER_FILES if (folder / f"{n}.csv").is_file()]
    if present and len(present) != len(OPTIMISER_FILES):
        absent = sorted(set(OPTIMISER_FILES) - set(present))
        raise DashboardError(f"{folder} has some optimiser files but not {', '.join(absent)}")
    tables = {n: read_csv(folder / f"{n}.csv", n) for n in (*VIEW_FILES, *present)}
    if not tables["channel_daily"]:
        raise DashboardError(f"{folder / 'channel_daily.csv'} has no rows")
    return tables


def _ratio(a, b):
    return None if not b else a / b


def summary(tables: dict[str, list[dict]]) -> dict:
    """The figures the page shows, from the extracts alone."""
    daily = tables["channel_daily"]
    measures = ("spend", "impressions", "clicks", "sessions", "purchases", "revenue")
    channels: dict[str, dict] = {}
    monthly: dict[tuple, dict] = defaultdict(lambda: dict.fromkeys(("spend", "revenue"), 0.0))
    for r in daily:
        c = channels.setdefault(
            r["channel"], dict(channel=r["channel"], channel_group=r["channel_group"],
                               **dict.fromkeys(measures, 0.0))
        )  # fmt: skip
        for m in measures:
            c[m] += r[m]
        cell = monthly[(r["month"], r["channel"])]
        cell["spend"] += r["spend"]
        cell["revenue"] += r["revenue"]
    for c in channels.values():
        c.update(
            ctr=_ratio(c["clicks"], c["impressions"]),
            cpc=_ratio(c["spend"], c["clicks"]),
            conversion_rate=_ratio(c["purchases"], c["clicks"]),
            cost_per_purchase=_ratio(c["spend"], c["purchases"]),
            roas=_ratio(c["revenue"], c["spend"]),
        )

    groups: dict[str, dict] = {}
    for r in tables["acquisition_monthly"]:
        g = groups.setdefault(r["channel_group"], dict(channel_group=r["channel_group"],
                              paid=False, spend=0.0, new_customers=0.0))  # fmt: skip
        g["paid"] = g["paid"] or r["paid"]
        g["spend"] += r["spend"]
        g["new_customers"] += r["new_customers"]
    for g in groups.values():
        g["cac"] = _ratio(g["spend"], g["new_customers"]) if g["paid"] else None

    totals = dict.fromkeys(COLUMNS["blended_monthly"], 0.0)
    del totals["month"]
    for r in tables["blended_monthly"]:
        for k in totals:
            totals[k] += r[k]
    kpis = dict(
        totals,
        blended_roas=_ratio(totals["revenue"], totals["spend"]),
        blended_roi=_ratio(totals["revenue"] - totals["spend"], totals["spend"]),
        paid_roas=_ratio(totals["paid_revenue"], totals["spend"]),
        blended_cac=_ratio(totals["spend"], totals["new_customers"]),
        first_day=min(r["date"] for r in daily),
        last_day=max(r["date"] for r in daily),
    )

    # Retention, pooled over the cohorts whose month is complete, as the analysis
    # report pools it: customers buying that month over the cohorts' customers.
    pooled: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    matrix: dict[tuple, float] = {}
    for r in tables["cohorts"]:
        k = int(r["months_since"])
        if r["complete"] and k <= COHORT_MONTHS:
            matrix[(r["cohort_month"], k)] = r["active_customers"] / r["cohort_size"]
            if k >= 1:
                pooled[k][0] += r["active_customers"]
                pooled[k][1] += r["cohort_size"]
    retention = {k: a / n for k, (a, n) in sorted(pooled.items())}

    out = dict(
        kpis=kpis,
        channels=sorted(channels.values(), key=lambda c: (-c["spend"], c["channel"])),
        monthly={k: dict(v) for k, v in sorted(monthly.items())},
        acquisition=sorted(groups.values(),
                           key=lambda g: (not g["paid"], -g["new_customers"], g["channel_group"])),
        retention=retention,
        cohort_matrix=matrix,
        ltv=tables["ltv_by_channel"],
        optimiser=None,
    )  # fmt: skip
    if "budget_gains" in tables:
        out["optimiser"] = dict(
            splits=tables["budget_splits"],
            gains=tables["budget_gains"],
            holdout=tables["holdout_lift"],
        )
    return out


def _label(split: str) -> str:
    """A split's name as the README writes it: "prior sd 0.5 + lift", "true optimum"."""
    if split == "true_optimum":
        return "true optimum"
    return split.replace("prior_", "prior sd ").replace("+", " + ")


def figures(s: dict) -> list[tuple[str, go.Figure]]:
    """One figure per sheet of the spec, in its order."""
    out = []
    names = {c["channel"]: c["channel_group"] for c in s["channels"]}
    order = [c["channel"] for c in s["channels"]]

    months = sorted({m for m, _ in s["monthly"]})
    colour = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}
    fig = go.Figure()
    for c in order:
        fig.add_bar(x=months, y=[s["monthly"].get((m, c), {}).get("spend", 0.0) for m in months],
                    name=f"{names[c]} spend", marker_color=colour[c])  # fmt: skip
    for c in order:
        fig.add_scatter(x=months,
                        y=[s["monthly"].get((m, c), {}).get("revenue", 0.0) for m in months],
                        name=f"{names[c]} revenue (last click)", mode="lines+markers",
                        line=dict(dash="dot", color=colour[c]))  # fmt: skip
    fig.update_layout(barmode="stack", yaxis_title="dollars a month",
                      legend=dict(orientation="h", y=-0.2))  # fmt: skip
    out.append(("Spend and last-click revenue by month", fig))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROAS (revenue / spend)",
                                                        "Cost per purchase"))  # fmt: skip
    labels = [names[c] for c in order]
    fig.add_bar(x=labels, y=[c["roas"] for c in s["channels"]], name="ROAS", row=1, col=1,
                text=[_fmt(c["roas"], "{:.2f}") for c in s["channels"]],
                marker_color=[colour[c] for c in order])  # fmt: skip
    fig.add_bar(x=labels, y=[c["cost_per_purchase"] for c in s["channels"]],
                name="cost per purchase", row=1, col=2,
                text=[_fmt(c["cost_per_purchase"], "${:,.0f}") for c in s["channels"]],
                marker_color=[colour[c] for c in order])  # fmt: skip
    fig.update_layout(showlegend=False)
    out.append(("Channel performance, by last click", fig))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("New customers", "CAC (paid groups)"))
    acq = s["acquisition"]
    fig.add_bar(x=[g["channel_group"] for g in acq], y=[g["new_customers"] for g in acq],
                name="new customers", row=1, col=1)  # fmt: skip
    paid = [g for g in acq if g["paid"]]
    fig.add_bar(x=[g["channel_group"] for g in paid], y=[g["cac"] for g in paid], name="CAC",
                row=1, col=2, text=[_fmt(g["cac"], "${:,.0f}") for g in paid])  # fmt: skip
    fig.update_layout(showlegend=False)
    out.append(("Customer acquisition, by the group credited with the first purchase", fig))

    cohorts = sorted({m for m, _ in s["cohort_matrix"]})
    ks = list(range(1, COHORT_MONTHS + 1))
    z = [[s["cohort_matrix"].get((m, k)) for k in ks] for m in cohorts]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=("Share of each cohort buying, by month after the first",
                                        "Pooled over complete cohorts"))  # fmt: skip
    fig.add_heatmap(x=ks, y=[m.isoformat()[:7] for m in cohorts], z=z, colorscale="Blues",
                    colorbar=dict(tickformat=".0%", x=0.6), row=1, col=1)  # fmt: skip
    fig.add_scatter(x=list(s["retention"]), y=list(s["retention"].values()),
                    mode="lines+markers", name="retention", row=1, col=2)  # fmt: skip
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_layout(showlegend=False)
    out.append(("Cohort retention", fig))

    ltv = [r for r in s["ltv"] if r["paid"]]
    fig = go.Figure()
    if not ltv:
        # The file has a row only for groups with a customer followed for the whole window.
        fig.add_annotation(text="No paid customer is followed for the whole window: the data "
                           "is shorter than it", showarrow=False)  # fmt: skip
        out.append(("Customer value against CAC", fig))
    window = int(ltv[0]["window_days"]) if ltv else None
    fig.add_bar(x=[r["channel_group"] for r in ltv], y=[r["revenue_per_customer"] for r in ltv],
                name=f"revenue per customer, first {window} days", marker_color=PROJECTED,
                text=[_fmt(r["revenue_per_customer"], "${:,.0f}") for r in ltv])  # fmt: skip
    fig.add_bar(
        x=[r["channel_group"] for r in ltv],
        y=[r["cac"] for r in ltv],
        name="CAC",
        marker_color=TRUE,
        text=[_fmt(r["cac"], "${:,.0f}") for r in ltv],
    )
    fig.update_layout(barmode="group", yaxis_title="dollars per customer",
                      legend=dict(orientation="h", y=-0.2))  # fmt: skip
    if ltv:
        out.append((f"Customer value against CAC (first {window} days)", fig))

    o = s["optimiser"]
    if o is None:
        return out
    shown = {"current": CURRENT, "recommendation": PROJECTED, "true_optimum": TRUE}
    fig = go.Figure()
    for split, split_colour in shown.items():
        rows = {r["channel"]: r for r in o["splits"] if r["split"] == split}
        fig.add_bar(x=[names.get(c, c) for c in order],
                    y=[rows[c]["spend_per_day"] if c in rows else None for c in order],
                    name=_label(split), marker_color=split_colour)  # fmt: skip
    fig.update_layout(barmode="group", yaxis_title="dollars a day",
                      legend=dict(orientation="h", y=-0.2))  # fmt: skip
    out.append(("Daily budget: current, recommended, and the generator's optimum", fig))

    gains = [g for g in o["gains"] if g["kind"] != "current"]
    fig = go.Figure()
    fig.add_bar(
        x=[_label(g["split"]) for g in gains],
        y=[g["projected_gain_share"] for g in gains],
        name="projected by the fit (the recommendation: the least any fit projects)",
        marker_color=PROJECTED,
    )
    fig.add_bar(
        x=[_label(g["split"]) for g in gains],
        y=[g["true_gain_share"] for g in gains],
        name="true, from the generator's curves (not measurable on real data)",
        marker_color=TRUE,
    )
    fig.update_layout(barmode="group", yaxis=dict(title="share of new-customer revenue a day",
                      tickformat=".0%"), legend=dict(orientation="h", y=-0.3))  # fmt: skip
    out.append(("Gain over the current budgets spent steadily: projected against true", fig))

    hold = o["holdout"]
    fig = go.Figure()
    with_own = [h for h in hold if h["own_projected_lift"] is not None]
    fig.add_bar(
        x=[_label(h["split"]) for h in with_own], y=[h["own_projected_lift"] for h in with_own],
        name="projected (bar: the split's own fit, or for the recommendation the least; "
        "whiskers: least to most over all fits)",
        marker_color=PROJECTED,
        error_y=dict(type="data", symmetric=False,
                     array=[h["most_projected_lift"] - h["own_projected_lift"] for h in with_own],
                     arrayminus=[h["own_projected_lift"] - h["least_projected_lift"]
                                 for h in with_own]),
    )  # fmt: skip
    fig.add_bar(x=[_label(h["split"]) for h in hold], y=[h["true_lift"] for h in hold],
                name="true, from the generator's curves", marker_color=TRUE)  # fmt: skip
    first = hold[0]["first_day"] if hold else None
    fig.update_layout(barmode="group", yaxis_title="new-customer revenue over the quarter",
                      legend=dict(orientation="h", y=-0.3))  # fmt: skip
    out.append((f"Held-out quarter from {first}: lift from each split", fig))
    return out


def _fmt(value, pattern: str) -> str:
    return "n/a" if value is None else pattern.format(value)


def page(s: dict, *, cdn: bool = False) -> str:
    """The whole HTML page."""
    k = s["kpis"]
    tiles = [
        ("Spend", _fmt(k["spend"], "${:,.0f}")),
        ("Revenue", _fmt(k["revenue"], "${:,.0f}")),
        ("Blended ROAS", _fmt(k["blended_roas"], "{:.2f}")),
        ("ROI on revenue", _fmt(k["blended_roi"], "{:.1%}")),
        ("New customers", _fmt(k["new_customers"], "{:,.0f}")),
        ("Blended CAC", _fmt(k["blended_cac"], "${:,.2f}")),
    ]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Marketing ROI dashboard</title><style>",
        "body{font-family:system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px;"
        "color:#1f2933;background:#fff}",
        ".note{background:#fff8e1;border-left:4px solid #f0b400;padding:8px 12px;margin:8px 0}",
        ".tiles{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}",
        ".tile{border:1px solid #d9e2ec;border-radius:6px;padding:10px 14px;min-width:140px}",
        ".tile b{display:block;font-size:1.4em}h2{margin-top:32px;font-size:1.15em}",
        "</style></head><body>",
        "<h1>Marketing ROI dashboard</h1>",
        f"<p>{k['first_day']} to {k['last_day']}. Drawn from the CSV extracts with Plotly "
        f"{html.escape(plotly.__version__)}, as specified in docs/dashboard-spec.md.</p>",
        "<p class='note'><b>The data is synthetic</b>: made by this project's generator, "
        "not collected from any real company, campaign or GA4 property.</p>",
        "<p class='note'>Revenue and purchases are credited by <b>last click</b>, as GA4 "
        "records them, which overstates paid search and understates display and video. "
        "ROI is on revenue, (revenue - spend) / spend: there is no cost of goods.</p>",
        "<div class='tiles'>",
        *(f"<div class='tile'>{html.escape(t)}<b>{html.escape(v)}</b></div>" for t, v in tiles),
        "</div>",
    ]
    if s["optimiser"] is not None:
        parts.append(
            "<p class='note'><b>Projected is not measured.</b> A projected gain or lift is a "
            "fitted model's forecast. The true figures come from the generator's own curves, "
            "which only generated data has; no real analysis could show them.</p>"
        )
    first = True
    for title, fig in figures(s):
        fig.update_layout(template="plotly_white", margin=dict(t=60, l=60, r=20, b=60),
                          height=460)  # fmt: skip
        parts.append(f"<h2>{html.escape(title)}</h2>")
        include = ("cdn" if cdn else True) if first else False
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include,
                                 config={"displaylogo": False}))  # fmt: skip
        first = False
    if s["optimiser"] is None:
        parts.append("<p>No optimiser results in the extracts, so the budget charts are left "
                     "out: write them with <code>--optimise</code>.</p>")  # fmt: skip
    parts.append("</body></html>")
    return "\n".join(parts)


def write(s: dict, path: Path, *, cdn: bool = False) -> int:
    """Write the page to ``path``; its size in bytes."""
    path = Path(path)
    text = page(s, cdn=cdn)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise DashboardError(f"cannot write {path}: {exc.strerror or exc}") from exc
    return len(text.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m marketing_roi.dashboard",
        description="Draw the dashboard from the CSV extracts as one HTML page.",
    )
    parser.add_argument("--extracts", type=Path, required=True,
                        help="the folder marketing_roi.extracts wrote")  # fmt: skip
    parser.add_argument("--out", type=Path, required=True, help="the HTML file to write")
    parser.add_argument("--cdn", action="store_true",
                        help="link Plotly's JavaScript from its CDN, not include it")  # fmt: skip
    args = parser.parse_args(argv)
    if args.out.is_dir():
        parser.error(f"--out {args.out} is a folder; name the HTML file to write")
    try:
        size = write(summary(read(args.extracts)), args.out, cdn=args.cdn)
    except DashboardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
