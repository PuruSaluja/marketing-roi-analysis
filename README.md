# Marketing Campaign ROI Analysis

Channel performance and return on ad spend, analysed in SQL and Python, and a
marketing mix model that estimates how each channel's spend turns into
revenue over time, used to propose a better split of a fixed budget.

> **Status:** project skeleton. The package, its tests and CI are in place;
> the data, the analysis and the model are not built yet, so there are no
> results here. Each one added will come from a run of the committed code.

## The data is synthetic

**Every number this project analyses is generated, not collected.** There is
no real company, campaign, customer or Google Analytics property behind it.
A seeded generator in this repository will produce:

- daily spend, impressions, clicks, conversions and revenue per channel, at
  cost-per-acquisition and conversion rates drawn to sit within published
  industry benchmarks, which will be cited beside the generator;
- a session and attribution table modelled on Google Analytics 4 data, with
  each session's traffic source, campaign and conversions.

The generator also decides the true answer: how long each channel's effect
carries over (adstock), where its returns flatten (saturation), the baseline
and the seasonality. That is the point of using it. A model fitted to the
data can be checked against the curves that actually made it, which real
data never allows, and the README will report how closely it recovers them.
It also means the results describe this generator, not any real market.

**Projected is not measured.** The budget optimiser will report the revenue
its reallocation is *projected* to gain under the fitted model, checked on a
holdout period the model never saw. That is a model's forecast, not an
observed lift, and it will be labelled as one wherever it appears.

## What it will contain

| Part | |
| --- | --- |
| Generator | Channels, spend, funnel and revenue, and the GA4-shaped table |
| PostgreSQL | Schema and indexes, loaded from the generator's output |
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

## License

MIT — see [LICENSE](LICENSE).
