# CO2 Trace: audited carbon accounting, 1990-2022

![ci](https://github.com/pro-nine/CO2-Trace-Global/actions/workflows/ci.yml/badge.svg)

A reproducible analysis of 162 countries' CO2 on the OWID snapshot committed in `data/`. The first version wrote its findings as fixed text. This version recomputes every number, tests the earlier claims against the data, and reports which ones survive.

## What the audit found
| v1 claim | verdict | why |
|---|---|---|
| Germany clusters with Ukraine, not Denmark and Sweden | not supported | Germany is 0.53 from Ukraine but 0.21 from Denmark and 0.11 from Sweden on log-trajectory distance; Germany co-clusters with Denmark and Sweden in 100% of bootstrap draws and with Ukraine in 46% |
| Most reducers cut CO2 mainly through efficiency | supported | 82% of 39 reducers; 78% to 85% under 3% input noise |
| Absolute decoupling never lasts (longest 7 years) | definition-sensitive | strictly consecutive spells: 70% last one year and the longest is 7; allowing one interrupted year, the longest is 16 |
| 41 countries show offshoring | misleading | 24 of the 41 have rising CO2 per capita and a median GDP per capita growth of 131% |
| Decomposition covers all 162 countries | denominator differs | 154 at both endpoints, 153 in every year |

Full text, tables and figures: `outputs/analysis_report.md`, `outputs/tables/`, `outputs/figures/`.

## Methods
- **Data contract and integrity:** `data/panel.manifest.json` pins the snapshot's SHA-256; `load_panel` refuses a mismatch or a contract violation. Refreshing from OWID is an explicit flag.
- **Chain-linked LMDI:** year-by-year additive Kaya decomposition that closes exactly, compared with the two-endpoint version. A noise-sensitivity analysis shows how stable each country's dominant driver is.
- **Tapio decoupling and survival analysis:** eight elasticity states, spells of strong decoupling, and Kaplan-Meier curves with right-censoring and a log-rank test by income group.
- **Trajectory similarity:** RMS distance on log trajectories, hierarchical clustering, and moving-block bootstrap co-assignment instead of single-run K-means.
- **Synthetic control:** donors screened on the pre-policy period, fit scored on held-out pre-policy years, in-space placebo inference, and Japan 2011 as a positive control.

## Run
```
uv sync
uv run python -m scripts.run_analysis          # about a minute
uv run python -m scripts.run_analysis --fast   # no placebo inference
uv run pytest -q
```

## Limits
- CO2 is territorial. Consumption-based emissions and trade flows are not in the data.
- Kaya LMDI is accounting. The efficiency effect mixes technology, structural change and offshoring, and the panel has no sector split.
- Synthetic control is exploratory. Cases and years were picked by the analyst and point estimates were seen before the report was written. Japan 2011 reaches only p = 0.10, so the test has low power here and non-significant results are not evidence of no effect.
- Noise ranges are sensitivity analyses, not sampling confidence intervals.
