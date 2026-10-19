# Dashboard spec

A one-page dashboard of the last-click analysis and the budget optimiser, to be
built in Tableau from the CSV extracts. [`dashboard.py`](../src/marketing_roi/dashboard.py)
draws the same sheets with Plotly from the same files; that page is what the
tests check, figure by figure, against the analysis views.

**This spec has not been built in Tableau.** No Tableau was available, so no
`.twb` or `.twbx` workbook is in the repository and the calculated fields below,
written in Tableau's calculation language, have not been run there. They are
the formulas the Plotly page computes.

## Data sources

```bash
python -m marketing_roi.extracts --out extracts --optimise results/optimise-42.json
```

Connect each file with Tableau's text file connector as its own data source;
no sheet needs a join. Dates are `YYYY-MM-DD`, booleans `true`/`false`, and an
empty field is NULL. Set `month` and `cohort_month` to dates at month level.

| File | One row per | Dimensions | Measures |
| --- | --- | --- | --- |
| `channel_daily.csv` | day and paid channel | date, month, channel, channel_group | spend, impressions, clicks, sessions, purchases, revenue (last click) |
| `acquisition_monthly.csv` | month and channel group | month, channel_group, paid | spend, new_customers |
| `blended_monthly.csv` | month | month | spend, purchases, revenue, new_customers, new_customer_revenue, paid_purchases, paid_revenue |
| `cohorts.csv` | first-purchase month and month after it | cohort_month, months_since, month, complete | cohort_size, active_customers, purchases, revenue, cumulative_purchases, cumulative_revenue |
| `ltv_by_channel.csv` | channel group | window_days, channel_group, paid | customers, spend, cac, revenue_per_customer, ltv_to_cac |
| `budget_splits.csv` | split and channel | split, kind, channel, channel_group | spend_per_day, current_per_day |
| `budget_gains.csv` | split | split, kind | projected_gain_per_day, projected_gain_share, true_gain_per_day, true_gain_share, new_customer_revenue_per_day, holdout_mape |
| `holdout_lift.csv` | split | split, kind, first_day | days, quarter_spend, quarter_new_customer_revenue, own_projected_lift, least_projected_lift, most_projected_lift, true_lift |

The first four files hold only measures that add up: take every ratio over
sums, never an average of rows. `ltv_by_channel.csv` and the three optimiser
files hold one row per item, already final; aggregate them with `ATTR` or
`MIN` and never across rows. `cohort_size` repeats on every row of a cohort,
so it is summed only within one `months_since`. `kind` is `current`, `fit`,
`recommendation` or `true optimum`; `split` is a fit's name (`committed`,
`prior_0.5+lift`, ...), `current`, `recommendation` or `true_optimum`.

## Calculated fields

Each is NULL where its denominator is zero, as the views' ratios are.

| Field | Source | Formula |
| --- | --- | --- |
| CTR | channel_daily | `IF SUM([impressions]) > 0 THEN SUM([clicks]) / SUM([impressions]) END` |
| CPC | channel_daily | `IF SUM([clicks]) > 0 THEN SUM([spend]) / SUM([clicks]) END` |
| Purchases per click | channel_daily | `IF SUM([clicks]) > 0 THEN SUM([purchases]) / SUM([clicks]) END` |
| Cost per purchase | channel_daily | `IF SUM([purchases]) > 0 THEN SUM([spend]) / SUM([purchases]) END` |
| ROAS | channel_daily | `IF SUM([spend]) > 0 THEN SUM([revenue]) / SUM([spend]) END` |
| CAC | acquisition_monthly | `IF MAX(INT([paid])) = 1 AND SUM([new_customers]) > 0 THEN SUM([spend]) / SUM([new_customers]) END` |
| Blended ROAS | blended_monthly | `SUM([revenue]) / SUM([spend])` |
| ROI on revenue | blended_monthly | `(SUM([revenue]) - SUM([spend])) / SUM([spend])` |
| Blended CAC | blended_monthly | `SUM([spend]) / SUM([new_customers])` |
| Retention | cohorts | `SUM([active_customers]) / SUM([cohort_size])` |
| Revenue per customer to date | cohorts | `SUM([cumulative_revenue]) / SUM([cohort_size])` |
| Reallocation | budget_splits | `SUM([spend_per_day]) - SUM([current_per_day])` |
| ROI lift, points | holdout_lift | `100 * ATTR([true_lift]) / ATTR([quarter_spend])` |

ROI is on revenue, (revenue - spend) / spend; the data has no cost of goods,
so it is not a profit ROI. Retention is read only over `complete = true`, so a
month the data ends inside of does not show as a drop.

## Sheets

Every sheet that shows purchases or revenue carries the caption "credited by
last click". Every optimiser sheet shows projected and true figures side by
side in fixed colours (projected blue, true red) and names them so: a
**projected** gain is a fitted model's forecast; a **true** gain comes from the
generator's curves and could not be known on real data.

| # | Sheet | Source | Columns / rows | Marks | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Spend and last-click revenue by month | channel_daily | MONTH(month) / SUM(spend) and SUM(revenue) on one axis | stacked bars for spend, dotted lines for revenue; colour = channel_group | one colour per channel on both marks |
| 2 | Channel performance | channel_daily | channel_group / ROAS, Cost per purchase | bars, labels on | sorted by SUM(spend), descending |
| 3 | Customer acquisition | acquisition_monthly | channel_group / SUM(new_customers); CAC for paid groups | bars | unpaid groups have no CAC |
| 4 | Cohort retention | cohorts | months_since 1-12 / cohort_month; colour = Retention | heat map, plus a line of Retention by months_since pooled over cohorts | filter complete = true |
| 5 | Customer value against CAC | ltv_by_channel | channel_group (paid) / revenue_per_customer, cac | grouped bars | title gives window_days; empty if the data is shorter than the window |
| 6 | Daily budget | budget_splits | channel_group / SUM(spend_per_day) | grouped bars; colour = split, for current, recommendation, true_optimum | tooltip: Reallocation |
| 7 | Gain, projected against true | budget_gains | split / projected_gain_share, true_gain_share | grouped bars, percent axis | the recommendation's projection is the least any fit projects for it |
| 8 | Held-out quarter | holdout_lift | split / own_projected_lift, true_lift | grouped bars; error bars least_projected_lift to most_projected_lift | title gives first_day; tooltip: ROI lift, points |

## Dashboard

From the top, at 1,200 px wide:

1. Title, period (MIN and MAX of channel_daily's date), and two banners: "The
   data is synthetic" and the last-click note above.
2. Six tiles from blended_monthly: spend, revenue, Blended ROAS, ROI on revenue,
   new customers, Blended CAC.
3. Sheets 1 to 5, full width each (sheets 2, 3 and 4 each two panels side by side).
4. A banner "Projected is not measured", then sheets 6 to 8.

Filters: a month range on channel_daily, acquisition_monthly and
blended_monthly (shared through a `month` parameter, since they are separate
sources), and a channel group filter on sheets 1, 2 and 6. The optimiser
sheets describe one run's training and held-out days and take no date filter.

## The Plotly reproduction

```bash
python -m marketing_roi.dashboard --extracts extracts --out dashboard.html   # --cdn for a small file
```

It draws sheets 1 to 8 in this order from the files above, computing the
fields they plot as written here (not the two tooltip fields, Reallocation and
ROI lift), as a static page without the filters. Without the optimiser files it draws sheets
1 to 5 and says the rest are left out. Its tests load a generated 91-day run,
write the extracts, and check every figure on the page against the views
(channel totals and ratios, monthly spend and revenue, acquisition and CAC,
the blended totals, pooled and per-cohort retention, to a relative 1e-9), and
that each chart draws those figures.
