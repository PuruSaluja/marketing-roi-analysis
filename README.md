# Marketing Campaign ROI Analysis

Channel performance and return on ad spend, analysed in SQL and Python, and a
marketing mix model that estimates how each channel's spend turns into
revenue over time, used to propose a better split of a fixed budget.

> **Status:** the data generator, the PostgreSQL load, the last-click
> analysis in SQL and the mix model's adstock transforms are built. The rest
> of the mix model and the optimiser are not built yet. Every figure here
> comes from a run of the committed code.

## The data is synthetic

**Every number this project analyses is generated, not collected.** There is
no real company, campaign, customer or Google Analytics property behind it.
A seeded generator in this repository produces:

- daily spend, impressions, clicks, purchases and revenue per paid channel,
  with cost per click, click-through and conversion rates taken from published
  benchmarks where one was found and marked as assumed where not (see
  [Generated data](#generated-data));
- a session and attribution table modelled on Google Analytics 4 data, with
  each session's traffic source, campaign and purchase.

The generator also decides the true answer: how long each channel's effect
carries over (adstock), where its returns flatten (saturation), the baseline
and the seasonality. That is the point of using it. A model fitted to the
data can be checked against the curves that actually made it, which real
data never allows, and the README will report how closely it recovers them.
The generator also knows which channel caused each purchase, so the
last-click credit in the GA4-style table can be compared with the truth.
It also means the results describe this generator, not any real market.

**Projected is not measured.** The budget optimiser will report the revenue
its reallocation is *projected* to gain under the fitted model, checked on a
holdout period the model never saw. That is a model's forecast, not an
observed lift, and it will be labelled as one wherever it appears.

## Generated data

```bash
python -m marketing_roi.generate --out data/generated --seed 42
```

writes four files (not committed; `data/generated/` is ignored):

| File | One row per | What it holds |
| --- | --- | --- |
| `channel_daily.csv` | day and paid channel | spend, impressions, clicks, and the purchases and revenue the channel's spend **caused**, including carryover from earlier days |
| `ga4_sessions.csv` | website session | source, medium, campaign, default channel group, device, engaged flag, and the purchase in the session, if any, credited to the **last click** |
| `truth_daily.csv` | day and component | each channel's adstocked spend and expected purchases, the baseline, and repeat purchases |
| `truth.json` | run | the settings, each channel's true adstock decay, saturation point and scale, and the rates the run produced |

The model behind it, in [`generate.py`](src/marketing_roi/generate.py):
demand is a trend times yearly seasonality (peaking in early December), a
weekly pattern and a Black Friday to Cyber Monday boost. Each channel's spend
follows a quarterly budget level and partly follows demand; display and video
run in flights of several weeks on and off. The purchases a channel causes are
Poisson around a Hill curve of its geometrically adstocked spend, scaled so
its purchases per click over the period match its conversion rate. New
customers no channel caused come from the baseline, and any purchase can be
followed by a repeat from the same customer, after a gap averaging 75 days.
Last-click credit is drawn so that search and direct pick up purchases that
display, video and social caused, as they do in practice.

Before each purchase the customer makes a few earlier visits (Poisson, mean
1.5, 1 to 14 days before), taken from that day's sessions without a purchase,
so the path to a purchase is in the table and a customer's `user_pseudo_id`
carries the time of their first visit. Other sessions come from a pool of
visitor ids, some of whom visit more than once.

**Where the rates come from** ([`channels.py`](src/marketing_roi/channels.py)):

| Channel | Mean spend a day | CPC | CTR | Conversion rate | Source |
| --- | --- | --- | --- | --- | --- |
| Paid search | $900 | $3.49 | 8.92% | 3.83% | [WordStream Google Ads Benchmarks 2025](https://www.wordstream.com/blog/2025-google-ads-benchmarks), Shopping, Collectibles & Gifts |
| Paid social | $600 | $0.34 | 4.13% | 0.5% | CPC and CTR: [WordStream Facebook Ads Benchmarks 2025](https://www.wordstream.com/blog/facebook-ads-benchmarks-2025), traffic campaigns, same industry; conversion rate **assumed** |
| Display | $300 | $2.64 | 0.25% | 1.33% | [LocaliQ Display & Video Ads Benchmarks 2026](https://localiq.com/blog/display-video-ads-benchmarks/), Apparel / Fashion & Jewelry; conversion rate set from its cost per lead (below) |
| Online video | $400 | $1.00 | 0.7% | 0.6% | **assumed** |

WordStream's figures are medians across campaigns: 16,446 US search campaigns
on Google and Microsoft Ads from April 2024 to March 2025, and 554 US Facebook
traffic campaigns from April 2024 to June 2025, where an industry's row can
rest on as few as three. Its conversion rate counts any conversion an
advertiser tracked, a lead as well as a sale; the generator uses it as a
purchase rate. Medians do not divide into each other, so $3.49 / 3.83% gives a
cost per purchase of $91 while the industry's median cost per lead is $47.94.
The generator follows the click and conversion rates.

LocaliQ's display figures are medians over 7,074 US display campaigns from
April 2025 to June 2026. Its industries have no shopping row, so the nearest,
Apparel / Fashion & Jewelry, is used. It gives no conversion rate, so display's
is $2.64 / $198.34 = 1.33%, which makes its cost per conversion the published
median cost per lead. No source with a stated method was found for video, or
for a social purchase rate: those stay assumed. LocaliQ's only video figure is
a 0.09% CTR for video ads on the USA TODAY network, which is not YouTube
inventory and comes without a cost or conversion rate.

**The default run** (seed 42, 2024-01-01 to 2025-12-31, 731 days) wrote
2,924 channel-day rows, 2,428,372 sessions (a 257 MB CSV) and 40,057
purchases, in 19.4 to 20.4 s over three runs. Its realised rates:

| Channel | Spend | CTR | CPC | Conversion rate | Cost per purchase | First-order ROAS |
| --- | --- | --- | --- | --- | --- | --- |
| Paid search | $657,899.91 | 8.92% | $3.49 | 3.92% | $88.97 | 1.01 |
| Paid social | $438,600.03 | 4.14% | $0.34 | 0.50% | $67.76 | 1.32 |
| Display | $219,299.99 | 0.25% | $2.62 | 1.27% | $205.92 | 0.44 |
| Online video | $292,400.01 | 0.70% | $1.00 | 0.61% | $163.17 | 0.55 |

ROAS here counts only the purchase a channel caused, not the repeat purchases
that customer goes on to make. Of the 40,057 purchases, 16,725 were caused by
paid channels, 11,152 came from the baseline and 12,180 were repeats. Last
click credits paid search with 8,889 purchases against the 7,395 it caused,
display with 158 against 1,065, and video with 162 against 1,792. Display and
video get even fewer than their last-click shares (15% and 10%) because
purchases they cause while a flight is off, from carryover, have no session of
theirs to land on and are credited to direct (18 purchases in this run).
77.4% of the 27,877 customers have a visit before their first purchase.

## PostgreSQL

```bash
createdb marketing_roi
export MARKETING_ROI_DSN="host=localhost port=5432 user=postgres dbname=marketing_roi"
python -m marketing_roi.load --data data/generated     # --replace to load again
```

The connection string can be any form libpq accepts; the password can stay in
`PGPASSWORD` or a pgpass file. The load runs in one transaction: it copies
the files into staging tables, inserts typed rows, and checks the result
against `truth.json` before committing: row counts, and purchases and revenue
agreeing to the cent day by day and for each channel and day. A damaged file,
a renamed column or a mismatch leaves the database as it was. A second load
is refused unless `--replace` is given.

Two schemas, from [`schema.sql`](src/marketing_roi/sql/schema.sql):

| Table | |
| --- | --- |
| `marketing.channels` | the paid channels, with source, medium and channel group |
| `marketing.channel_daily` | spend, impressions, clicks, caused purchases and revenue per day and channel |
| `marketing.traffic_sources` | one row per source, medium and campaign, linked to its channel if paid |
| `marketing.ga4_sessions` | sessions with their source, device, engagement and purchase; `ga4_sessions_flat` joins the source back in |
| `truth.daily`, `truth.run` | what only the generator knows: each component's purchases, and `truth.json` itself |

`truth` is a separate schema so the analysis cannot read the answer by
accident. Checks hold the rules the generator follows: clicks never exceed
impressions, a purchase has a transaction id and revenue or neither, a
session with a purchase is engaged, and no amount is negative, NaN or
infinite.

Secondary indexes ([`indexes.sql`](src/marketing_roi/sql/indexes.sql)) are
built after the copy, then the tables are vacuumed and analysed:

| Index | Serves |
| --- | --- |
| `ga4_purchases_by_date` (partial, purchases only) | revenue and last-click credit by day and source |
| `ga4_purchases_by_user` (partial) | each customer's first purchase and repeats, for cohorts |
| `ga4_sessions_date_source` | traffic by day and source |
| `channel_daily_by_channel` | one channel's days in order |

Loading the default run (2,428,372 sessions) took 50.6 to 54.4 s over three
runs on the development machine; the database came to 315 MB. A month of
purchases by day and source then reads 1,985 rows from `ga4_purchases_by_date`
with an index-only scan and no table lookups, in 0.56 to 1.03 ms over five
runs. Without the vacuum the planner chose the transaction id's unique index
instead and the query took 90 to 166 ms: the table's visibility map was not
yet set, so an index-only scan would still have read the table. A plain
VACUUM, without ANALYZE, was enough to switch the plan.

## Last-click analysis in SQL

```bash
python -m marketing_roi.analysis              # --ltv-days 365 for a longer window
```

[`analysis.sql`](src/marketing_roi/sql/analysis.sql) creates an `analysis`
schema of views; the loader creates it at the end of each load, and the command
above recreates it and prints the whole period's figures. The views read only
what a business would have: spend, impressions and clicks as the ad platforms
report them, and purchases as GA4 records them, each credited to the last click
before it. They never read the `truth` schema or the purchases each channel
caused, and a test shows it: with `truth` dropped and the caused purchases set to
zero, every view returns the same rows. Every view is also checked against the
same figures worked out in Python from the generated files.

| View | One row per | What it gives |
| --- | --- | --- |
| `channel_performance` | paid channel and month, and the whole period | spend, impressions, clicks, sessions, purchases, revenue, CTR, CPC, purchases per click, cost per purchase, ROAS |
| `acquisition` | channel group and month, and the whole period | new customers by the channel their first purchase is credited to; CAC for paid groups |
| `blended` | month, and the whole period | all spend against all revenue and new customers: blended ROAS, ROI on revenue, CAC |
| `cohorts` | first-purchase month and each month after it | customers buying again, retention, cumulative revenue and purchases per customer, and whether the month is complete |
| `ltv_by_channel(days)` | channel group | revenue per customer over their first `days` days against CAC, counting only customers followed for the whole window |

Ratios are taken over sums (a period's spend over its clicks, never an average of
daily ratios) and are NULL where there is nothing to divide by, as for a
flighted channel's dark days. A row whose month is NULL covers the whole period.
ROI is on revenue, (revenue - spend) / spend; the data has no cost of goods, so it
is not a profit ROI.

**The default run, by last click** (2024-01-01 to 2025-12-31):

| Channel | Spend | Purchases | Purchases per click | Cost per purchase | ROAS |
| --- | --- | --- | --- | --- | --- |
| Paid search | $657,899.91 | 8,889 | 4.71% | $74.01 | 1.21 |
| Paid social | $438,600.03 | 2,887 | 0.22% | $151.92 | 0.59 |
| Online video | $292,400.01 | 162 | 0.06% | $1,804.94 | 0.05 |
| Display | $219,299.99 | 158 | 0.19% | $1,387.97 | 0.06 |

Blended over the two years: $1,608,199.94 of spend against $3,606,965.07 of
revenue, a blended ROAS of 2.24 and an ROI on revenue of 124.3%, with 27,877 new
customers at a blended CAC of $57.69. Of those customers, 8,889 were acquired
through paid search, 2,887 paid social, 162 video and 158 display; the rest came
through organic search (8,032), direct (6,641) and referral (1,108). Every
purchase last click gives to a paid channel is a first purchase, because the
generator sends repeat purchases' last clicks to direct, email and organic
search, so each paid channel's CAC equals its cost per purchase above.

Over the first 180 days, customers bring $123 to $134 each whatever channel
acquired them (the first order alone averages $89.96), so the channels differ by
what they cost, not by the customers they bring. For customers acquired by
2025-07-05, the last day whose 180 days the data covers:

| Acquired via | Customers | Revenue per customer | CAC | LTV:CAC |
| --- | --- | --- | --- | --- |
| Paid search | 6,474 | $127.99 | $73.88 | 1.73 |
| Paid social | 2,048 | $126.93 | $151.14 | 0.84 |
| Display | 115 | $134.41 | $1,398.34 | 0.10 |
| Online video | 111 | $123.33 | $1,784.73 | 0.07 |

Of each month's new customers, 10.2% buy again in the next calendar month, 6.0%
three months on, 2.7% six months on and 0.6% twelve months on (complete months
only, pooled over cohorts); the January to March 2024 cohorts reach $137 to $138
of revenue per customer twelve months on.

**This is the view last click gives, and it is wrong in a known way.** The
generator recorded what each channel actually caused; in the same run the cost
per purchase it caused was $88.97 for search, $67.76 for social, $163.17 for
video and $205.92 for display (7,395, 6,473, 1,792 and 1,065 purchases). Last
click overstates search by a fifth (8,889 against 7,395) and credits social with
45% of what it caused, display with 15% and video with 9%, because the purchases
they cause mostly end on a search, direct or organic visit. Reading the table above, a budget would move
out of display and video; the mix model is there to estimate what they cause
from spend and outcomes alone, without the truth.

Creating the views takes about 0.05 s. On the full load, each view reads in 0.1
to 0.7 s (cohorts the slowest, 0.41 to 0.66 s over five runs) and the whole
report in 1.4 to 1.5 s over three runs.

## Adstock

An ad seen today can bring a purchase next week, so the mix model will regress
outcomes not on each day's spend but on its *adstock*: a weighted sum of that
day's spend and the days before. [`adstock.py`](src/marketing_roi/adstock.py)
has the transforms:

| Transform | Weight on spend `l` days back | Shape of the effect |
| --- | --- | --- |
| `geometric_adstock(spend, decay)` | `decay ** l`, computed as `a[t] = spend[t] + decay * a[t-1]` | peaks on the day of spend, fades by the same share each day |
| `weibull_adstock(..., form="survival")` | `exp(-(l / scale) ** shape)`, over `max_lag` days | peaks on the day of spend; shape below 1 fades fast then lingers, above 1 holds then drops; shape 1 is geometric with `decay = exp(-1 / scale)` |
| `weibull_adstock(..., form="pmf")` | the Weibull probability of `[l, l + 1)` | with shape above 1, peaks days after the spend |

They normalise by default, dividing the weights by their sum, so a steady $100
a day has an adstock of $100 a day once the window has filled and the result
stays in dollars per day. Days before the data count as no spend; the geometric
form can instead treat them as the first day's spend repeated (`start="steady"`),
which is how the generator starts. `half_life`, `mean_lag` and `lag_for_share`
put a transform's carryover in days. The tests check each form against its
closed form (an impulse, a constant spend, linearity, shape 1 against geometric),
and that the geometric form reproduces the adstock the generator recorded for
every channel, to a relative 1e-12.

The generator's channels use geometric adstock, and these are the decays a fitted
model should recover:

| Channel | Decay | Half-life | Mean lag | Days until 90% of the effect is in |
| --- | --- | --- | --- | --- |
| Paid search | 0.10 | 0.30 days | 0.11 days | same day |
| Paid social | 0.40 | 0.76 days | 0.67 days | 2 days later |
| Display | 0.50 | 1.00 day | 1.00 day | 3 days later |
| Online video | 0.70 | 1.94 days | 2.33 days | 6 days later |

The carryover shows in the generator's record: on the 255 days display was dark
it still caused 37 purchases, and video 134 on its 235 dark days (of 1,065 and
1,792 in all).

## Parts

| Part | |
| --- | --- |
| Generator (built) | Channels, spend, funnel and revenue, and the GA4-shaped table |
| PostgreSQL (built) | Schema and indexes, loaded from the generator's output |
| Analysis SQL (built) | Channel performance, CAC, ROAS, blended ROI, cohorts, by last click |
| Marketing mix model | Adstock transforms (built); saturation curves per channel, regression with baseline and seasonality, fit reported on a holdout period |
| Optimiser | Reallocates a fixed total budget over the fitted response curves |
| Excel workbook | The SQL's logic rebuilt in formulas (openpyxl) |
| Dashboards | Tableau-ready extracts, a dashboard spec, and a Plotly reproduction |

## Development

Requires Python 3.11 or later.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Unix
pip install -c constraints.txt -e ".[dev]"   # the tested versions; drop -c for the newest
pytest
ruff check . && ruff format --check .
```

The database tests need a disposable PostgreSQL database whose name ends in
`_test`; they drop and recreate the two schemas in it:

```bash
createdb marketing_roi_test
export MARKETING_ROI_TEST_DSN="host=localhost port=5432 user=postgres dbname=marketing_roi_test"
```

Without `MARKETING_ROI_TEST_DSN` they are skipped. CI runs them against a
PostgreSQL 16 service and sets `MARKETING_ROI_REQUIRE_DB`, which turns that
skip into a failure. It runs the suite on Python 3.11 and 3.14 with the newest
versions `pyproject.toml` allows, and on 3.14 with the exact versions in
`constraints.txt`.

## License

MIT — see [LICENSE](LICENSE).
