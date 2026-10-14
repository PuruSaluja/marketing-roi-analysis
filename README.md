# Marketing Campaign ROI Analysis

Channel performance and return on ad spend, analysed in SQL and Python, and a
marketing mix model that estimates how each channel's spend turns into
revenue over time, used to propose a better split of a fixed budget.

> **Status:** the data generator and the PostgreSQL load are built. The
> analysis and the model are not built yet, so there are no results here.
> Each one added will come from a run of the committed code.

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

One simplification to know about for cohort work: a customer's id appears only
on the sessions in which they bought. Sessions without a purchase come from a
separate pool of visitor ids, so the table cannot show the visits that led up
to a first purchase.

**Where the rates come from** ([`channels.py`](src/marketing_roi/channels.py)):

| Channel | Mean spend a day | CPC | CTR | Conversion rate | Source |
| --- | --- | --- | --- | --- | --- |
| Paid search | $900 | $3.49 | 8.92% | 3.83% | [WordStream Google Ads Benchmarks 2025](https://www.wordstream.com/blog/2025-google-ads-benchmarks), Shopping, Collectibles & Gifts |
| Paid social | $600 | $0.34 | 4.13% | 0.5% | CPC and CTR: [WordStream Facebook Ads Benchmarks 2025](https://www.wordstream.com/blog/facebook-ads-benchmarks-2025), traffic campaigns, same industry; conversion rate **assumed** |
| Display | $300 | $0.60 | 0.5% | 0.4% | **assumed** |
| Online video | $400 | $1.00 | 0.7% | 0.6% | **assumed** |

WordStream's figures are medians across campaigns: 16,446 US search campaigns
on Google and Microsoft Ads from April 2024 to March 2025, and 554 US Facebook
traffic campaigns from April 2024 to June 2025, where an industry's row can
rest on as few as three. Its conversion rate counts any conversion an
advertiser tracked, a lead as well as a sale; the generator uses it as a
purchase rate. Medians do not divide into each other, so $3.49 / 3.83% gives a
cost per purchase of $91 while the industry's median cost per lead is $47.94.
The generator follows the click and conversion rates.

**The default run** (seed 42, 2024-01-01 to 2025-12-31, 731 days) wrote
2,924 channel-day rows, 2,676,740 sessions (a 282 MB CSV) and 40,449
purchases, in 17.7 to 21.0 s over three runs. Its realised rates:

| Channel | Spend | CTR | CPC | Conversion rate | Cost per purchase | First-order ROAS |
| --- | --- | --- | --- | --- | --- | --- |
| Paid search | $657,899.91 | 8.92% | $3.49 | 3.87% | $90.12 | 0.99 |
| Paid social | $438,600.03 | 4.14% | $0.34 | 0.50% | $68.10 | 1.33 |
| Display | $219,299.99 | 0.50% | $0.60 | 0.39% | $155.86 | 0.58 |
| Online video | $292,400.01 | 0.70% | $1.00 | 0.62% | $161.19 | 0.57 |

ROAS here counts only the purchase a channel caused, not the repeat purchases
that customer goes on to make. Of the 40,449 purchases, 16,962 were caused by
paid channels, 11,186 came from the baseline and 12,301 were repeats. Last
click credits paid search with 8,954 purchases against the 7,300 it caused,
display with 182 against 1,407, and video with 161 against 1,814. Display and
video get even fewer than their last-click shares (15% and 10%) because
purchases they cause while a flight is off, from carryover, have no session of
theirs to land on and are credited to direct (20 purchases in this run).

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

Loading the default run (2,676,740 sessions) took 54.0 to 57.1 s over three
runs on the development machine; the database came to 345 MB. A month of
purchases by day and source then reads 2,019 rows from `ga4_purchases_by_date`
with an index-only scan and no table lookups, in 0.52 to 1.07 ms over five
runs. Without the vacuum the planner chose the transaction id's unique index
instead and the query took 97 to 180 ms: the table's visibility map was not
yet set, so an index-only scan would still have read the table. A plain
VACUUM, without ANALYZE, was enough to switch the plan.

## Parts

| Part | |
| --- | --- |
| Generator (built) | Channels, spend, funnel and revenue, and the GA4-shaped table |
| PostgreSQL (built) | Schema and indexes, loaded from the generator's output |
| Analysis SQL | Channel performance, CAC, ROAS, blended ROI, cohorts |
| Marketing mix model | Adstock and saturation transforms per channel, regression with baseline and seasonality, fit reported on a holdout period |
| Optimiser | Reallocates a fixed total budget over the fitted response curves |
| Excel workbook | The SQL's logic rebuilt in formulas (openpyxl) |
| Dashboards | Tableau-ready extracts, a dashboard spec, and a Plotly reproduction |

## Development

Requires Python 3.11 or later.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Unix
pip install -e ".[dev]"
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
skip into a failure.

## License

MIT — see [LICENSE](LICENSE).
