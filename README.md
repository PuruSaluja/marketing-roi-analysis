# Marketing Campaign ROI Analysis

Channel performance and return on ad spend, analysed in SQL and Python, and a
marketing mix model that estimates how each channel's spend turns into
revenue over time, used to propose a better split of a fixed budget.

> **Status:** the data generator, the PostgreSQL load, the last-click
> analysis in SQL and the mix model (adstock, saturation, and a fit scored
> on held-out days) are built. The fit does not yet recover each channel's
> curve (see [Mix model fit](#mix-model-fit)); the optimiser is not built.
> Every figure here comes from a run of the committed code.

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
above recreates it and prints the whole period's figures. Recreating it is
refused if the schema holds anything the module did not create, since dropping
the schema would drop that too. The views read only what a business would have:
spend, impressions and clicks as the ad platforms report them, and purchases as
GA4 records them, each credited to the last click before it. They never read the
`truth` schema or the purchases each channel caused, and a test shows it: with
`truth` dropped and the caused purchases set to zero, every row of every view
stays the same. The figures in the five below are also checked against the same
figures worked out in Python from the generated files, on data that crosses a
year end.

A customer is a `user_pseudo_id`, and their first purchase in the data is taken
as their first. Both hold for the generator, which keeps one id per customer and
has no customers from before its first day. Neither holds for a real GA4 export:
its id is per browser or device, so a customer who buys on two devices counts
twice, and one who first bought before the data starts counts as new.

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
what they cost, not by the customers they bring. That is by construction: the
generator draws every order value from the same distribution and gives every
customer the same chance of buying again. For customers acquired by
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

Creating the views takes 0.05 to 0.08 s. On the full load, each view reads in
0.1 to 0.9 s over 15 runs in three sets (cohorts the slowest, 0.41 to 0.90 s),
and the whole report in 1.3 to 1.7 s over 11 runs.

## Adstock

An ad seen today can bring a purchase next week, so the mix model will regress
outcomes not on each day's spend but on its *adstock*: a weighted sum of that
day's spend and the days before. [`adstock.py`](src/marketing_roi/adstock.py)
has the transforms:

| Transform | Weight on spend `l` days back | Shape of the effect |
| --- | --- | --- |
| `geometric_adstock(spend, decay)` | `decay ** l`, computed as `a[t] = spend[t] + decay * a[t-1]` | peaks on the day of spend, fades by the same share each day |
| `weibull_adstock(..., form="survival")` | `exp(-(l / scale) ** shape)`, over `max_lag` days | peaks on the day of spend; shape below 1 fades fast then lingers, above 1 holds then drops; shape 1 is geometric with `decay = exp(-1 / scale)` |
| `weibull_adstock(..., form="pmf")` | the Weibull probability of `[l, l + 1)` | with shape above 1, can peak after the day of spend, near the Weibull's mode: shape 2.5 and scale 6 peak 4 days later, while with scale 1 they still peak on the day |

They normalise by default, dividing the weights by their sum, so a steady $100
a day has an adstock of $100 a day once the window has filled (the geometric
form's window never fills, so it approaches $100) and the result stays in
dollars per day. Parameters can be NumPy numbers, as a grid search over them
gives. Days before the data count as no spend; the geometric
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

## Saturation

A channel's first dollars reach the people most likely to buy and later ones
reach fewer new buyers, so the mix model passes each channel's adstocked
spend through a curve that flattens.
[`saturation.py`](src/marketing_roi/saturation.py) has two:

| Curve | Formula | Shape |
| --- | --- | --- |
| `hill(x, half_saturation, shape)` | `x^shape / (x^shape + half_saturation^shape)` | rises to a ceiling of 1, half of it at `half_saturation`; an S when `shape` is above 1, steepest at `half_saturation * ((shape - 1) / (shape + 1))^(1 / shape)` |
| `log_response(x, scale)` | `ln(1 + x / scale)` | no ceiling; each extra dollar does `1 / (scale + x)` |

`ResponseCurve` multiplies either by `beta`, giving purchases a day at an
adstocked daily spend, and has the slope (`marginal`), the spend per purchase
(`average_cost`) and the cost of one more purchase (`marginal_cost`, one over
the slope). At a steady budget the adstock equals the spend, so the curve is
also the steady daily response to a daily budget. The Hill curve is computed
as `1 / (1 + (half_saturation / x)^shape)`, so it stays finite where
`x^shape` alone overflows; the tests check the slopes against numerical
derivatives and that `true_curves` reproduces the generator's expected
purchases for every channel, to a relative 1e-12.

These are the curves the generator used for the default run (seed 42), the
truth a fitted model should recover. Costs are at a steady spend of the
channel's mean daily budget:

| Channel | Half-saturation | Shape | Ceiling (purchases a day) | Purchases a day at the mean | Cost per purchase | Cost of the next purchase | ... at half the budget | ... at 1.5 times |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paid search ($900 a day) | $720 | 1.2 | 17.76 | 10.06 | $89.46 | $171.98 | $91.37 | $291.20 |
| Paid social ($600) | $720 | 1.5 | 20.91 | 9.03 | $66.43 | $77.97 | $57.27 | $118.04 |
| Display ($300) | $300 | 1.3 | 3.65 | 1.83 | $164.34 | $252.83 | $153.86 | $406.20 |
| Online video ($400) | $800 | 1.0 | 8.21 | 2.74 | $146.12 | $219.17 | $152.20 | $298.32 |

The next purchase costs more than the average one on every channel, most on
paid search (nearly twice), which is already past the steep part of its
curve; paid social's next purchase costs least, so a fixed budget would do
more there. The optimiser will make that comparison on the *fitted* curves.
The ceiling is `beta`, which the generator sets so each channel's expected
purchases over the run equal its conversion rate times the clicks its spend
buys at its cost per click: an expected cost per purchase of cost per click
over conversion rate (paid search $91.12, paid social $68.00, display
$198.34, video $166.67). The steady figure above is lower on every channel:
spend varies from day to day (display and video run in flights), and these
curves get fewer purchases from uneven spend than from the same total spent
evenly. The realised cost per purchase adds chance (paid search $88.97,
display $205.92).

## Mix model fit

[`mmm.py`](src/marketing_roi/mmm.py) reads only what a business would see,
each channel's daily spend and the shop's daily revenue in GA4, and models
revenue on day `t` as a baseline plus, for each channel,
`beta * hill(adstock(spend))`. The baseline has an intercept, a linear trend,
two yearly sine/cosine pairs, a weekday pattern and the Black Friday to Cyber
Monday window. Each channel has a geometric decay, a half-saturation and a
shape; given those the model is linear, so the betas and baseline come from
least squares, with every beta held at 0 or above. The curves' parameters are
searched for, on a coarse grid one channel at a time and then with a
Nelder-Mead simplex over all twelve, within bounds (decay 0-0.95,
half-saturation 0.1-10 times the channel's mean daily spend, shape 0.5-4).
The search sees only the training days: the last 91 days (2025-10-02 to
2025-12-31, the holiday quarter) are held out, and the tests check that
changing them leaves the fit as it was.

```bash
python -m marketing_roi.mmm                  # new-customer revenue (default)
python -m marketing_roi.mmm --target total   # all revenue
```

The revenue modelled is **new-customer revenue**, each GA4 user id's first
purchase. In the generator paid media causes first purchases, and repeat
purchases follow from earlier customers whatever brought them, so this is
what spend moves directly. It also avoids a start-up artefact: the generator
begins with no customers, so on the default run repeat revenue averages $662
a day over the first quarter and climbs to $1,723 by the fourth, where it
levels off ($1,657 to $1,897 a day in the quarters after); a linear trend
fitted through the climb carries on past it and overshoots the held-out
quarter (by 6.6% on the default run, by 7.3% with the true curves). That
target was chosen after fitting all revenue on the default run and seeds 1
to 3; seeds 10 to 19 below were run afterwards. Both targets are reported.
MAPE leaves out any day without revenue (none of the runs here has one).

Three comparisons put the scores in context: the **same regression given the
true curves** (the generator's decays, half-saturations and shapes, with
only the betas and baseline fitted); the **baseline terms alone**, which the
media terms have to beat; and, for new-customer revenue, the **truth's
expected revenue**, each day's expected first purchases from the generator
times the mean order value, which no model of spend can improve on except by
chance. On the default run (seed 42):

| Revenue | Model | Train R² | Train MAPE | Held-out R² | Held-out MAPE | Held-out RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| New customers | fitted | 0.385 | 15.1% | 0.474 | 14.3% | $706 |
| New customers | true curves | 0.375 | 15.2% | 0.399 | 15.0% | $755 |
| New customers | baseline only | 0.295 | 16.3% | 0.267 | 15.6% | $834 |
| New customers | truth's expected revenue | 0.360 | 15.2% | 0.433 | 14.5% | $733 |
| All | fitted | 0.459 | 12.7% | 0.139 | 14.8% | $972 |
| All | true curves | 0.443 | 12.9% | 0.133 | 15.1% | $975 |
| All | baseline only | 0.403 | 13.3% | 0.112 | 14.9% | $987 |

The R² values are low because most of a day's revenue variation is chance:
purchases are Poisson draws and order values vary by about half their mean,
so even the truth's expected revenue explains only 0.433 of the held-out
variation (0.360 on the training days) with a MAPE of 14.5%. The fitted
model is at that floor; its slightly better held-out figures are chance too.
Over ten more seeds (10 to 19), modelling new-customer revenue:

| Model | Held-out R², min / median / max | Held-out MAPE, min / median / max |
| --- | --- | --- |
| Fitted | -0.048 / 0.160 / 0.257 | 12.7% / 16.5% / 18.3% |
| True curves | -0.060 / 0.188 / 0.339 | 12.8% / 15.6% / 18.6% |
| Baseline only | -0.773 / -0.040 / 0.229 | 14.0% / 18.1% / 23.6% |

The fitted model's held-out MAPE was below the baseline's on all ten seeds
and at most 1.3 points above the true curves' (below them on three), and its
training error was never above theirs, so the search does not stop short of
the fit the true curves give. Modelling all revenue instead, the fitted model
beat the baseline's held-out MAPE on 7 of the 10, and on one (seed 10) was
10.4 points worse than the true curves, with a held-out R² of -1.32.

**The fit does not recover each channel's curve.** On the default run it
puts paid search's shape at the upper bound (4.0, true 1.2) and credits paid
search with $143,875 of new-customer revenue over the training days, a
quarter of the $573,010 it caused; display's decay comes out 0.00 (true
0.50). Over the ten seeds the revenue credited to paid search ran from 0.13
to 56.8 times what it caused (within 25% on 1 of 10), and to paid social
from 0.38 to 8.5 times (2 of 10); 19 of the 40 fitted shapes sat at the
upper bound. Given the true curves, the same regression credits paid search
within 25% on 6 of 10 seeds and paid social on 9 of 10, so with the curves
known the regression comes much closer; it is the curves that two years of
noisy daily data do not pin down. Paid search and paid social run every day
with little variation (their daily spend varies by 25% and 30% of its mean,
display's and video's, which run in flights, by about 80%), and part of it
moves with the seasons (the baseline terms explain 20% of paid search's daily
spend over the training days and 60% of paid social's), which leaves little
independent variation to learn their curves from. Accurate held-out totals are therefore no evidence that the
per-channel curves are right, and a budget optimiser should not be run on
these curves as they stand.

## Can the channel curves be recovered?

[`identify.py`](src/marketing_roi/identify.py) fits generated runs several
ways and checks each channel's fitted curve against the generator's, by two
ratios (1 is right): the revenue the fit **credits** to the channel over the
training days against the revenue it caused, and the fitted **marginal**
return (extra revenue a day from one more dollar a day at the channel's mean
training spend) against the true one, the figure a reallocation equalises.
The ways tried, all options of `mmm.fit`:

- **committed**: the fit above.
- **prior sd**: log-normal priors on every shape (centre 1, a curve without
  an S) and half-saturation (centre the channel's mean training spend), with
  the log's standard deviation 0.25, 0.5 or 1. The noise variance that weighs
  them against the squared error is the baseline terms' residual variance.
- **weekly**: weekly sums of revenue and of each channel's daily response.
- **lift**: a measured lift for paid search, standing in for a holdout
  experiment: over 28 days of the training period (from 2024-07-01), the
  revenue paid search really caused there times 1 + e, e drawn from
  N(0, 0.15), with a stated standard error of 15%. It enters the regression
  as one more row on paid search's coefficient. This uses the generator's
  truth, as a real experiment would measure it; it is not free information.
- **true curves**: the regression given the generator's curves, its ceiling.

```bash
python -m marketing_roi.identify --seeds 10-19 --workers 5 --json results/identify-dev-10-19.json
python -m marketing_roi.identify --seeds 20-29 --workers 5 --json results/identify-validation-20-29.json
python -m marketing_roi.identify --seeds 30-34 --moved-truth --workers 5 --json results/identify-moved-30-34.json
```

The rule was written down before seeds 20 to 29 were first run: choose one
way on seeds 10 to 19 (the most channel-seed pairs with both ratios within
25%, held-out MAPE no more than a point worse than the committed fit's), then
on seeds 20 to 29 count a channel's curve as recovered if both ratios are
within 25% on at least 7 of the 10. On seeds 10 to 19 prior sd 0.25 + lift and
prior sd 0.5 + lift tied at 12 of 40 pairs; the weaker prior, sd 0.5, was
chosen. The figures, in [`results/`](results), on seeds 20 to 29 (seeds with
both ratios within 25%, of 10, per channel):

| Fit | Paid search | Paid social | Display | Video | Pairs, of 40 | Median held-out MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| committed | 0 | 0 | 2 | 0 | 2 | 15.4% |
| prior sd 0.25 | 7 | 0 | 5 | 4 | 16 | 14.8% |
| prior sd 0.5 | 7 | 0 | 3 | 4 | 14 | 14.8% |
| prior sd 1 | 6 | 0 | 3 | 2 | 11 | 14.8% |
| weekly | 1 | 0 | 0 | 0 | 1 | 15.3% |
| lift | 4 | 0 | 2 | 0 | 6 | 15.5% |
| prior sd 0.25 + lift | 8 | 0 | 5 | 4 | 17 | 14.5% |
| **prior sd 0.5 + lift (chosen)** | **6** | **0** | **3** | **4** | **13** | **14.5%** |
| prior sd 1 + lift | 5 | 0 | 4 | 1 | 10 | 14.8% |
| true curves | 6 | 4 | 5 | 7 | 22 | 14.4% |

**By that rule no channel's curve is recovered.** The chosen fit comes
closest on paid search, 6 of 10 (credit within 25% on 9, marginal on 6),
which is a fail. It does narrow the committed fit's errors a great deal: the
committed fit's credit ran from 0.13 to 106 times the truth for paid search
and 0.17 to 411 for paid social; the chosen fit's from 0.72 to 1.24 and 0.80
to 2.29. Even given the true curves the regression gets both ratios within
25% on 7 of 10 seeds only for video, so the bar is above what two years of
noisy daily data allow for three of the four channels.

**The priors help here because their centre is near this generator's
truth** (shapes 1.0 to 1.5, half-saturations 0.8 to 2 times mean spend), not
because they draw anything from the data: under them the fitted shapes sit
at the centre (medians 0.99 to 1.01 per channel for the chosen fit on seeds
20 to 29). With
every true shape moved to 2.5 and half-saturation to 2.5 times the daily
spend (seeds 30 to 34, `--moved-truth`), the order turns over: of 20 pairs,
committed 13, lift 15, prior sd 0.5 + lift 7, prior sd 0.25 none. The chosen
fit then puts display's shape at 1.00 to 1.31 and its marginal return at
0.39 to 0.73 of the truth. The lift helps the channel it measures and no
other: paid search's credit came within 25% on 9 of 10 seeds with it and 0
of 10 without, the other channels' counts barely moved. Weekly sums did worse than daily
fitting on all three sets of seeds (1, 3 and 6 pairs against 2, 4 and 13).

So the curves cannot be taken as known. A reallocation computed from one fit
would carry that fit's errors as if they were facts; the optimiser has to
show how far the answer moves across fits that the data cannot tell apart.

## Budget reallocation

[`optimise.py`](src/marketing_roi/optimise.py) keeps the total daily budget
as it is (each channel's mean spend over the training days, $2,146 a day on
the default run) and splits it again to maximise the expected revenue from
the fitted curves. At a steady daily spend a channel's adstock equals its
spend, so the curve gives the steady revenue a day at that budget; the split
treats display and video as running every day at their mean, and says
nothing about flighting. **Every gain below is measured from the current
budgets spent steadily**, which is not how they were spent (flights,
quarterly levels, daily noise). On the true curves that change alone is
worth $92 a day on the default run ($2,092 a day steadily against $1,999 as
really spent) and $72 to $92 on seeds 20 to 29 (a median 2.4% of
new-customer revenue), before any money moves between channels; the
optimiser reports it with each run. Each channel stays between half and twice its
current budget (`--floor`, `--cap`), where the curves were fitted. A Hill
curve with a shape above 1 is an S, on which equal marginal returns can mark
a minimum rather than the best split (a test shows one), so the split is found
exactly on a grid of 4,000 parts of the total ($0.54 a day here) by dynamic
programming over the channels.

```bash
python -m marketing_roi.optimise                                   # the default run, every fit
python -m marketing_roi.optimise --seeds 20-29 --workers 5         # ten more runs
python -m marketing_roi.optimise --seeds 20-29 --floor 0 --cap 5   # wider bounds
```

It optimises under each of the nine fits above that use no truth beyond the
measured lift, and reports them all. The **projected** gain is what a fit
expects from its own split; because the data are generated, each split is
also scored on the true curves, the **true** gain no real analysis could
measure. On the default run (dollars a day; [`results/optimise-42.json`](results/optimise-42.json)):

| Fit | Paid search | Paid social | Display | Video | Projected gain | True gain |
| --- | --- | --- | --- | --- | --- | --- |
| current, spent steadily | 877 | 584 | 310 | 375 | | |
| committed | 439 | 1,169 | 155 | 383 | +607 | +82 |
| prior sd 0.5 | 688 | 1,069 | 167 | 222 | +184 | +163 |
| weekly | 634 | 1,169 | 155 | 188 | +1,301 | +153 |
| lift | 634 | 1,169 | 155 | 188 | +522 | +153 |
| prior sd 0.5 + lift | 679 | 1,072 | 168 | 227 | +184 | +161 |
| true curves' optimum | 762 | 1,041 | 155 | 188 | | +168 |

(The other four fits are in the file.) All nine fits move money into paid
social and out of paid search and display, and eight of nine out of video,
which is the direction of the true optimum. How much they expect it to earn
differs sevenfold, $181 to $1,301 a day, while the true gains run from $82 to
$165. Over seeds 20 to 29 ([`results/optimise-20-29.json`](results/optimise-20-29.json)),
as medians over the runs, in shares of the run's new-customer revenue a
training day:

| Fit | Projected gain | True gain | Runs with a true loss | Share of the best possible gain |
| --- | --- | --- | --- | --- |
| committed | +5.7% | +1.3% | 3 of 10 | 26% |
| prior sd 0.25 | +3.9% | +4.4% | 1 | 93% |
| prior sd 0.5 | +3.8% | +4.4% | 1 | 92% |
| prior sd 1 | +3.6% | +4.1% | 1 | 84% |
| weekly | +10.5% | +3.4% | 2 | 62% |
| lift | +4.3% | +1.5% | 4 | 30% |
| prior sd 0.25 + lift | +3.3% | +4.4% | 0 | 92% |
| prior sd 0.5 + lift | +3.5% | +4.4% | 0 | 92% |
| prior sd 1 + lift | +3.5% | +4.1% | 1 | 89% |
| true curves' optimum | | +4.9% | | |

The best any split within the bounds could do is about 5% more new-customer
revenue than the same budgets spent steadily ($148 to $188 a day on these
runs), on top of the median 2.4% that spending them steadily earns. The committed fit expects more
than that (5.7%), gets a quarter of it, and loses money on three runs; its
splits point each channel the right way on 28 of the 40 channel-runs, the
prior sd 0.5 + lift fit's on 37. With the bounds widened to 0 to 5 times the
current budget ([`results/optimise-20-29-wide.json`](results/optimise-20-29-wide.json)) the committed fit projects
+29.8% and really loses 15.9%, on 8 of the 10 runs a loss; the fits with
priors change little (+4.3% true for prior sd 0.5 + lift; prior sd 1 most,
from +4.1% to +3.7%). The bounds are
what keep a fit's errors from turning into a loss.

**The fits with priors do best here for the reason found above: their centre
is close to this generator's truth.** Run on generated data with every
channel's curve away from that centre (seeds 30 to 39, on which the optimiser
had not been run), their advantage goes. The test was fixed before the runs:
the chosen fit (prior sd 0.5 + lift) had to get at least 80% of the best
possible gain, at least the committed fit's share, and a loss on at most 1 of
the 10 runs.

```bash
python -m marketing_roi.optimise --seeds 30-39 --truth moved --workers 5   # shape 2.5, half-saturation 2.5x spend
python -m marketing_roi.optimise --seeds 30-39 --truth low --workers 5     # shape 0.7, half-saturation 0.5x
```

| Truth | Best possible gain | Committed: true gain, losses, share of best | Prior sd 0.5 + lift: true gain, losses, share of best |
| --- | --- | --- | --- |
| moved | +45.3% ($1,347-$1,654 a day) | +37.0%, 0, 85% | +34.5%, 0, 78% |
| low | +0.7% ($18-$28 a day) | -0.9%, 9, -114% | +0.2%, 3, 27% |

(Medians over the ten runs; [`results/optimise-30-39-moved.json`](results/optimise-30-39-moved.json),
[`results/optimise-30-39-low.json`](results/optimise-30-39-low.json).) It fails on both. With
the moved truth (an S that saturates late) there is a great deal to gain, the
true optimum moves money into paid social on every run, and every fit makes
money; the committed fit gets more of it than the chosen one. With the low
truth (curves that saturate early) there is almost nothing to gain, and the
fits' errors are larger than the gain: the committed fit loses money on 9 of
the 10 runs, the chosen fit on 3. Spending steadily is worth -$198 to -$279 a
day under the moved truth (flights beat a steady spend on an S that has not
yet turned) and +$76 to +$94 under the low one, so "spend steadily" is not a
safe default either. The optimiser never answers "leave the budget as it
is", which is the right answer on the low truth. The direction is the steadier part
of the answer: on every one of seeds 20 to 29 the true optimum moves money
into paid social and out of the other three, and the fits' splits point the
same way on 311 of the 360 fit-channel-runs, though all nine agree on only 19
of the 40 channel-runs, and paid search is the least settled (most fits move
it up on seeds 24 and 26). It is less steady when the truth changes: the
fits point the true optimum's way on 288 of 360 fit-channel-runs under the
moved truth and 228 under the low one. The size of the gain is a projection,
and the fits' projections disagree with each other and with the truth.

## Parts

| Part | |
| --- | --- |
| Generator (built) | Channels, spend, funnel and revenue, and the GA4-shaped table |
| PostgreSQL (built) | Schema and indexes, loaded from the generator's output |
| Analysis SQL (built) | Channel performance, CAC, ROAS, blended ROI, cohorts, by last click |
| Marketing mix model (built) | Adstock, saturation curves, and a regression with baseline and seasonality, scored on a held-out quarter |
| Optimiser (built) | Reallocates a fixed total budget over the fitted response curves, under every fit, and shows the spread |
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
