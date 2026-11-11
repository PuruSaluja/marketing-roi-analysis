# Marketing Campaign ROI Analysis

Channel performance and return on ad spend, analysed in SQL and Python, and a
marketing mix model that estimates how each channel's spend turns into
revenue over time, used to propose a better split of a fixed budget.

> **Status:** complete. Every figure here comes from a run of the committed
> code; [`docs/findings.md`](docs/findings.md) is the findings report, with
> recommendations.

## Results

On two years of generated data for one online shop with four paid channels
(details below; the data is synthetic):

- **Last click misreads the channels.** It credits paid search with 8,889
  purchases against the 7,395 it caused, and display and video with 15% and
  9% of theirs; paid social is the cheapest channel per purchase caused
  ($67.76), where last click puts it at twice paid search's cost ($151.92
  against $74.01). See
  [Last-click analysis in SQL](docs/guide.md#last-click-analysis-in-sql).
- **A mix model forecasts total revenue at the floor chance allows, but does
  not recover each channel's curve**: held-out MAPE 14.3% against 14.5% for
  the truth's own expected revenue, while the revenue it credits to paid
  search ran from 0.13 to 56.8 times the truth over ten runs. Priors help only
  near their centre. See [Mix model fit](docs/guide.md#mix-model-fit) and
  [Can the channel curves be recovered?](docs/guide.md#can-the-channel-curves-be-recovered).
- **The direction of a reallocation is steadier than its size.** On the
  default run all nine fits move money into paid social and out of paid search
  and display, and eight of nine out of video, as the true optimum does; over
  ten more runs they point each channel its way on 311 of 360 fit-channel-runs.
  Their projected gains differ sevenfold. See
  [Budget reallocation](docs/guide.md#budget-reallocation).
- **The recommended split's true lift on a held-out quarter is +3.8% of
  new-customer revenue (+6.6 ROI points), with a loss on none of ten runs.**
  Its least projection had a median of +2.8%. The models' forecast error is as large as
  the lift, so measuring it on real data needs a holdout experiment. See
  [On the held-out quarter](docs/guide.md#on-the-held-out-quarter).

The same analysis is in an [Excel workbook](docs/guide.md#excel-workbook) of formulas,
checked against the SQL, and in [dashboards](docs/guide.md#dashboards): CSV extracts for
Tableau, a dashboard spec, and a Plotly page drawn from the extracts.

## The data is synthetic

**Every number this project analyses is generated, not collected.** There is
no real company, campaign, customer or Google Analytics property behind it. A
seeded generator produces two years of daily spend, clicks, purchases and
revenue for four paid channels, with rates taken from published benchmarks
where one was found and marked as assumed where not, and a session table
modelled on Google Analytics 4 data.

The generator also records the true answer: how long each channel's effect
carries over, where its returns flatten, and which channel caused each
purchase. That is the point of using it. A model fitted to the data can be
checked against the curves that made it, which real data never allows, and
last-click credit can be compared with what each channel caused. It also
means the results describe this generator, not any real market.

**Projected is not measured.** A gain from the budget optimiser is a fitted
model's forecast; the *true* gain, from the generator's own curves, is given
beside it and is a figure no real analysis could know. No lift here is
measured: no budget was changed in any market.

## Method

| Step | Module |
| --- | --- |
| 1. [Generate](docs/guide.md#generated-data) two years of daily channel data and GA4-style sessions, with the truth recorded beside them | [`generate.py`](src/marketing_roi/generate.py) |
| 2. [Load](docs/guide.md#postgresql) them into PostgreSQL in one checked transaction | [`load.py`](src/marketing_roi/load.py) |
| 3. [Analyse by last click](docs/guide.md#last-click-analysis-in-sql) in SQL views: channel performance, CAC, blended ROI, cohorts, LTV | [`analysis.sql`](src/marketing_roi/sql/analysis.sql) |
| 4. Rebuild the views in an [Excel workbook](docs/guide.md#excel-workbook) of formulas | [`workbook.py`](src/marketing_roi/workbook.py) |
| 5. [Model revenue](docs/guide.md#mix-model-fit) on adstocked, saturating spend, scored on a held-out quarter | [`mmm.py`](src/marketing_roi/mmm.py) |
| 6. [Test whether the channel curves can be recovered](docs/guide.md#can-the-channel-curves-be-recovered), with pass marks fixed before fresh runs | [`identify.py`](src/marketing_roi/identify.py) |
| 7. [Reallocate the budget](docs/guide.md#budget-reallocation) under every fit and try the recommendation on the held-out quarter | [`optimise.py`](src/marketing_roi/optimise.py) |
| 8. Write [extracts for Tableau, a dashboard spec and a Plotly dashboard](docs/guide.md#dashboards) | [`dashboard.py`](src/marketing_roi/dashboard.py) |

## Run it

Needs Python 3.11 or later and PostgreSQL 16.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Unix
pip install -c constraints.txt -e ".[dev]"
createdb marketing_roi
export MARKETING_ROI_DSN="host=localhost port=5432 user=postgres dbname=marketing_roi"
python -m marketing_roi.generate --out data/generated --seed 42
python -m marketing_roi.load --data data/generated
python -m marketing_roi.analysis
python -m marketing_roi.mmm
python -m marketing_roi.optimise --json results/optimise-42.json
python -m marketing_roi.extracts --out extracts --optimise results/optimise-42.json
python -m marketing_roi.dashboard --extracts extracts --out dashboard.html
```

The [guide](docs/guide.md#reproducing-the-results) gives the full sequence,
the workbook step and the commands behind every file in [`results/`](results).
`pytest` runs the suite; its database tests need a disposable database named
in `MARKETING_ROI_TEST_DSN` (see [Development](docs/guide.md#development)).

## More

- [Findings and recommendations](docs/findings.md), the report a marketing
  team would read.
- [The full guide](docs/guide.md): the data, the SQL, the workbook, adstock
  and saturation, the fit, curve recovery, the optimiser and the dashboards,
  each with its method and results.
- [Dashboard spec](docs/dashboard-spec.md) for the Tableau version.

## License

MIT — see [LICENSE](LICENSE).
