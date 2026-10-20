# Findings and recommendations

What the analysis and the mix model show about where this advertiser's budget
should go, and how sure each answer is. The figures are from runs of the
committed code; the [README](../README.md) gives the methods, the commands and
the full tables.

**The data is synthetic.** A seeded generator made every row: two years of
daily spend, clicks and GA4-style sessions for one online shop with four paid
channels (paid search, paid social, display and online video). It also
recorded what each channel's spend truly caused, which no real advertiser can
see. That makes three kinds of figure possible here, and this report keeps
them apart:

| Kind | What it is | Would a real analysis have it? |
| --- | --- | --- |
| **Measured** | read from the data a business has: platform spend and clicks, GA4's last-click purchases, a model's error on days it did not see | yes |
| **Projected** | what a fitted model expects a change to earn | yes, as a forecast |
| **True** | what the generator's own curves say a change earns | no; only generated data has it |

A real report could give only the first two. The third is here to show how
far the second can be trusted.

## Summary

1. **Last click gives paid search too much credit and display and video almost
   none.** It credits paid search with 8,889 purchases against the 7,395 it
   caused, paid social with 45% of what it caused, display with 15% and video
   with 9%. Read naively, their last-click ROAS of 0.05 and 0.06 would end
   display and video; what they caused cost $163 to $206 a purchase, not
   $1,388 to $1,805.
2. **Paid social is the cheapest channel per purchase caused ($67.76), where
   last click puts it at twice paid search's cost ($151.92 against $74.01).**
   Customers bring about the same revenue whichever channel acquired them
   ($123 to $134 in their first 180 days), so the channels differ in what
   they cost, not in whom they bring.
3. **A mix model forecasts total revenue as well as the truth allows, but
   cannot tell the channels' curves apart** from two years of daily data. Its
   held-out error is at the floor set by chance; the revenue it credits to
   paid search ran from 0.13 to 56.8 times what paid search caused across ten
   runs.
4. **The direction of a reallocation is steadier than its size.** On the
   default run all nine fits move money into paid social and out of paid
   search and display, and eight of nine out of video, as the true optimum
   does; over runs 20 to 29 they point each channel the true optimum's way on
   311 of 360 fit-channel-runs. How much the fits expect that to earn differs
   sevenfold.
5. **The recommended split truly adds 3.8% to new-customer revenue on a
   held-out quarter** (median over ten runs, 6.6 points of ROI, a loss on
   none), but that lift is no larger than the model's forecast error, so
   comparing revenue before and after could not show it. It has to be
   measured with a holdout experiment.

## 1. Last click misallocates credit

On the default run (2024-2025, $1,608,200 of spend, $3,606,965 of revenue,
blended ROAS 2.24), by last click and by what each channel's spend caused:

| Channel | Spend | Purchases, last click | Purchases caused | Cost per purchase, last click | Cost per purchase caused |
| --- | --- | --- | --- | --- | --- |
| Paid search | $657,900 | 8,889 | 7,395 | $74.01 | $88.97 |
| Paid social | $438,600 | 2,887 | 6,473 | $151.92 | $67.76 |
| Online video | $292,400 | 162 | 1,792 | $1,804.94 | $163.17 |
| Display | $219,300 | 158 | 1,065 | $1,387.97 | $205.92 |

The upper-funnel channels' purchases mostly end on a search, direct or
organic visit, so last click hands their credit on. The generator was built
to do that, so its size here is an assumption of the data, not a finding
about any market. The point that carries over is the method: a channel's
last-click ROAS is not evidence of what cutting it would cost.

## 2. Customers are worth the same; channels differ in cost

Revenue per customer over their first 180 days is $127.99 for paid search,
$126.93 for paid social, $134.41 for display and $123.33 for video (customers
acquired by 2025-07-05, credited by last click). The generator draws every
order from the same distribution, so this, too, is by construction. Against
CAC, only paid search pays back within 180 days by last click (LTV:CAC 1.73;
paid social 0.84). Blended, the shop acquires a customer for $57.69, because
57% of its 27,877 new customers come through organic search, direct and
referral, at no media cost.

## 3. The mix model fits totals, not channels

The model regresses daily new-customer revenue on each channel's adstocked,
saturating spend plus trend, seasonality and a Black Friday term, fitted on
the first 640 days and scored on the last 91. On the default run its held-out
MAPE is 14.3%, against 14.5% for the truth's own expected revenue: the
remaining error is chance in which customers buy and how much. Over ten more
runs its held-out MAPE beat a baseline without media on all ten.

Fitting totals well did not make the channel curves right. Paid search and
paid social run every day with little variation, partly in step with the
seasons, which leaves little independent movement to learn their curves from.
Priors on the curves' shapes and half-saturation points narrowed the errors
a great deal, but only because their centre sat near this generator's
truth; with the truth moved away from it, the fits with priors did worse
than the fit without them. No channel's curve met the recovery bar fixed
before the validation runs.

## 4. Reallocation: the direction holds, the size does not

Holding the total daily budget ($2,146 on the default run) and each channel
within half and twice its current budget, every one of nine fits moves money
into paid social and out of paid search and display, and eight of nine out of
video. Over runs 20 to 29 the true optimum does the same on every run, and
the fits point each channel its way on 311 of 360 fit-channel-runs.

The fits disagree on how much it is worth. On the default run they project
$181 to $1,301 more revenue a day (5.4% to 38.9% of new-customer revenue)
where the true gains run from $82 to $165. The committed fit, with neither
priors nor a lift test, projects +18.1% and truly earns +2.5%. One fit's
projection is not a forecast to plan on.

So the optimiser recommends from all the fits at once: the split whose
**smallest** projected gain over every fit is largest, and only if that is at
least 1% of new-customer revenue. On the default run that split moves paid
search from $877 to $650 a day (-26%), paid social from $584 to $1,087
(+86%), display from $310 to $160 (-48%) and video from $375 to $249 (-34%).
Where there was almost nothing to gain (runs with curves that saturate
early) it kept the budget on 9 of 10 runs, where the committed fit's split
lost money on 7.
On runs whose curves saturate late it took 49.5% of the best possible gain
against the 50% its pre-registered test asked for: it is cautious, and leaves
gain behind when there is a lot of it.

## 5. The held-out quarter

Applied to the last quarter, which no fit saw, with that quarter's flights and
daily pattern kept and only the shares between channels changed:

| | Default run | Runs 50 to 59, median |
| --- | --- | --- |
| Quarter's spend and new-customer ROI | $234,877; 55.5% | |
| Recommendation, least projected lift | $15,363 | +2.8% |
| Recommendation, true lift | **$14,330 (+3.9%), +6.1 ROI points** | **+3.8%, +6.6 points; 0 losses** |
| Best steady split, true lift | $16,395, +7.0 points | +5.1%, +8.4 points |
| The committed fit's own split, projected and true lift | $55,676 projected, $4,241 true | |

The recommendation changed the budget on 9 of the 10 runs and gained on all
9 (2.0% to 7.4% of the quarter's new-customer revenue, 3.7 to 10.4 ROI
points); on the tenth it kept the budget, where the prior sd 0.5 + lift
fit's own split would have lost 1.7%. These are **true** lifts: on real data they would be
unknown.

What a real analysis could see is the forecast. Each fit's forecast of the
quarter's revenue at the old spend missed by -12.9% to +7.5% over these runs
(median absolute error 2.6% to 4.1% by fit), as large as the 4% lift itself.
Revenue after a reallocation, compared with the model's forecast, would not
tell the lift from forecast error.

## Recommendations

1. **Shift budget toward paid social, within the bounds.** Take the robust
   split, not any one fit's: on the default run, paid social up 86% and paid
   search, display and video down 26%, 48% and 34%, total spend unchanged.
   Plan on a few percent of new-customer revenue, the least the fits project
   (median 2.8% on the held-out quarters), not the 5% to 39% that single
   fits project on the default run.
2. **Do not judge display and video by last-click ROAS.** It credits them
   with 9% to 15% of what they cause. Cutting them, as the robust split does,
   rests on the fits' marginal returns, which the held-out quarters bore out,
   not on last click.
3. **Measure the lift with an experiment before rolling it out.** Keep some
   regions or audiences on the current split for the quarter and compare
   like with like. A before-and-after read against the model cannot separate
   a 4% lift from a forecast error of the same size.
4. **Calibrate the model with lift tests on more channels.** One test on paid
   search brought the revenue the fit credits to it within 25% of the truth
   on 9 of runs 20 to 29, against 0 of 10 for the same fit without the test;
   the other channels barely moved. Paid social, where the money is moving,
   is the next to test.
5. **Report projections as forecasts, with their spread.** Show the range
   over fits that the data cannot tell apart, and the least of them as the
   planning figure. Keep the 1% bar, so that where there is little to gain
   the budget stays as it is.

## What this does not show

- **Anything about a real market.** The generator fixes the click and
  conversion rates (from published benchmarks where one was found, assumed
  where not), the curves, the carryover, how last click hands on credit, and
  that customers are worth the same from every channel. The findings show
  what this method recovers from data made that way.
- **Profit.** ROI is on revenue; the data has no cost of goods or margin.
- **Interactions.** Channels act independently in the generator: no channel
  raises another's response, and no competitor reacts.
- **Untested truths.** The recommendation was tested on three kinds of truth
  (the default, curves that saturate late and curves that saturate early). It
  failed its pre-registered test on the second by half a point; other truths
  might do worse.
