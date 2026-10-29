# Phillips Curve Flattening and Threshold Tests

An empirical study of U.S. inflation and labor-market tightness through 2026Q1.
The Python pipeline compares linear and kinked Phillips curves, estimates
rolling slopes and structural changes, and evaluates conditional projections
on a chronological holdout sample.

The hypothesis tested is that inflation responds more strongly to
labor-market tightness when the vacancy-to-unemployment ratio is high, so the
curve that looked flat before 2020 is nonlinear rather than absent.

![Inflation and labor-market tightness](figures/fig5_kink_scatter.png)

## Results

The saved results use a June 12, 2026 data vintage recorded with the original
local inputs. The long-sample regressions begin in 1960, and the rolling
60-quarter windows draw on data from 1955Q1. The vacancy-based models use 101
quarters from 2001Q1 to 2026Q1.

| Comparison | Saved estimate |
| --- | --- |
| Unemployment-gap slope, 1960–1989 vs. 1990–2019 | −0.5554 vs. −0.0011 |
| Quandt–Andrews slope-break date, 1960–2019 | 1983Q2; sup-Wald 16.21 vs. asymptotic 1% critical value 12.35; block-bootstrap p = 0.03 |
| Linear vacancy/unemployment slope, 2001–2026 | 1.106 (t = 4.14) |
| Estimated vacancy/unemployment kink | 1.18; the approximate LR search set spans the whole 0.40–1.40 grid |
| Slope above / slope below the kink | 3.59 |
| Bootstrap test of linearity | p = 0.21; linearity is not rejected |
| Holdout RMSE, trained through 2022Q4 (2023Q1–2026Q1), linear vs. kinked | 0.883 vs. 0.795 percentage points |
| Holdout RMSE, trained through 2019Q4 (2020Q1–2026Q1), linear vs. kinked | 1.808 vs. 1.814; the trained kink sits at the 0.40 grid edge |

A moving-block bootstrap of the stable-intercept-and-slope no-break model
puts the slope-break p-value at 0.0325 with the saved seed, significant at 5%
but not at 1%. This is an approximation under the maintained residual-process
assumptions. A difference from the asymptotic reference values does not itself
establish over-rejection or finite-sample size control. The earlier 300-seed
range is not a validation of the revised bootstrap implementation.

Both holdout comparisons re-estimate each model, including the threshold, on
the training quarters only, then project with realized expectations,
vacancies, unemployment, and supply pressure. These are conditional
projections using revised data, not real-time forecasts, and no test of
forecast superiority is implemented. Trained through 2022Q4, the kinked
model's RMSE over the 13 quarters from 2023Q1 to 2026Q1 is roughly 10% lower;
that difference is descriptive. Trained only on pre-2020 data, neither model
anticipates the 2021–2022 surge: from 2021Q2 to 2022Q4 both under-project
inflation by roughly 1.4 to 5 percentage points a quarter, and their RMSEs are
nearly equal.

In the fitted decomposition, cooling labor-market tightness accounts for
roughly 63% of the inflation decline between the 2022Q2–Q4 and
2025Q1–2026Q1 averages. This is model-based accounting, not a causal estimate,
and the linear model, which the bootstrap does not reject, attributes a
smaller share to tightness. Expectations proxies materially affect the
estimated curve shape: with Michigan household expectations, five of six
specifications put the kink near 0.43–0.44 and lose the steeper upper slope.

See [model estimates](results/results.json),
[decomposition](results/decomposition.csv), and
[2023–2026 holdout projections](figures/fig7_oos.png).

| Figure | Shows |
| --- | --- |
| [fig1_inflation_history.png](figures/fig1_inflation_history.png) | Headline and core CPI inflation with recession shading |
| [fig2_pc_scatter_eras.png](figures/fig2_pc_scatter_eras.png) | Inflation surprises against the unemployment gap, fitted by era |
| [fig3_rolling_slope.png](figures/fig3_rolling_slope.png) | Rolling 60-quarter unemployment-gap slope with a 90% band |
| [fig4_vu_timeseries.png](figures/fig4_vu_timeseries.png) | The vacancy/unemployment ratio against the estimated kink |
| [fig5_kink_scatter.png](figures/fig5_kink_scatter.png) | The supply-adjusted inflation gap against v/u with the kinked fit |
| [fig6_decomposition.png](figures/fig6_decomposition.png) | Fitted contributions to core inflation, 2020Q1–2026Q1 (kinked model) |
| [fig7_oos.png](figures/fig7_oos.png) | Projections of the linear and kinked models trained through 2022Q4, 2023Q1–2026Q1 |

## Run

Use Python 3.11. From this directory:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python code/04_tables.py
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate
it with `.venv\Scripts\activate`. The tests and table generation work
offline with the bundled results. Tests use synthetic data to check quarterly
transformations, run the full estimation on a planted kink, and validate
accounting, holdout metrics and the estimates quoted in the Results section.

For estimation and figure regeneration, obtain the source inputs first:

```sh
python code/00_fetch_data.py
sh code/run_all.sh
```

The downloader obtains the current source vintage and reuses existing valid
input files. Use `--force` only when deliberately refreshing them. A refresh
validates every requested series before it replaces any file, and it refuses
a download that covers a shorter history than the file it would replace.
Downloads and processed datasets are local and ignored by Git. Network access
is needed for downloading; no API key is required.

The full pipeline has been checked against the original local inputs and
reproduces the saved estimates. On October 29, 2026, live acquisition was checked for all 14 inputs; the offline tests cover input parsing and failed-download handling.

New downloads may revise historical values, so a fresh run can differ from
the bundled results. Exact numerical replication requires the original input
vintage, which is not included. The analysis end date stays fixed at 2026Q1;
refreshing inputs does not extend the study. Running the pipeline overwrites
the saved results and figures in the checkout. Regenerated PNG files can
differ byte for byte under another Pillow or zlib build even when every pixel
is identical.

The order is `01_build_dataset.py` → `02_analysis.py` → `03_figures.py` →
`04_tables.py`. Each script resolves paths from its own location. On Windows,
run those four scripts with `python` in that order. Estimation includes 499
linearity-bootstrap and 1,999 slope-break-bootstrap replications. Runtime
depends on the machine and numerical-library build.

### Offline demo on synthetic data

To run the estimation, figures, and tables without network access, use the
synthetic demo:

```sh
python code/run_demo.py
```

It writes synthetic series in the downloaded-input format to the ignored
`demo/` folder, copies the four pipeline scripts there, and runs them. The
saved results, figures, and any downloaded inputs are not touched. The
synthetic inflation gap has a kink planted at v/u = 1.0, with slopes of 0.5
below and 3.5 above, so the run shows whether the grid search recovers a
known threshold. With the default seed and the pinned packages it estimated
1.14, and the approximate LR search set (0.92–1.40) contained the planted
value. Figures and tables are explicitly labeled as synthetic; every number in
`demo/` is synthetic. Use `--prepare-only` to write the inputs
without estimating, and `--seed` to draw a different sample. Runtime depends on the machine. With `--check` it exits with an error unless the default run
recovers the planted kink; CI runs it this way.

## Data and method

| Input group | FRED series |
| --- | --- |
| Price indices | CPIAUCSL, CPILFESL, PCEPILFE, PCEPI, CPIENGSL |
| Labor market | UNRATE, NROU, JTSJOL, UNEMPLOY |
| Inflation expectations | MICH, EXPINF1YR, EXPINF10YR |
| Recession shading | USREC |

Series are downloaded from FRED's CSV endpoint and retain their source
identifiers. The [Michigan expectations series](https://fred.stlouisfed.org/series/MICH)
is attributed to Surveys of Consumers, University of Michigan, via FRED.
The [Global Supply Chain Pressure Index](https://www.newyorkfed.org/research/policy/gscpi)
comes from the Federal Reserve Bank of New York, and the 1-year and 10-year
expected inflation series (EXPINF1YR, EXPINF10YR) come from the Federal
Reserve Bank of Cleveland via FRED. Download provenance records
the URLs, retrieval time, and file checksums locally. Consult each source for
its usage terms.

Monthly observations are averaged within calendar quarters, and the build
step lists any quarter averaged from fewer than three months. The source files
have no October 2025 value for CPIAUCSL, CPILFESL, CPIENGSL, UNRATE or
UNEMPLOY, because BLS did not collect them during the federal shutdown
([CPI release](https://www.bls.gov/news.release/archives/cpi_12182025.htm),
[employment release](https://www.bls.gov/news.release/archives/empsit_12162025.htm)). Their
2025Q4 values therefore average November and December. Compared with
interpolating October, this raises annualized 2025Q4 inflation and lowers
2026Q1 inflation by about 0.37 percentage points for headline CPI and 0.28 for
core CPI. An isolated rerun on October 29, 2026 interpolated only those five missing
October values. The selected kink and bootstrap p-values were unchanged;
the 2023-2026 holdout RMSEs changed from 0.883/0.795 to 0.900/0.814 for the
linear/kinked models. The original observations remain the default.

Inflation is `400 × log(Pq / Pq−1)` for quarterly annualized rates and
`100 × log(Pq / Pq−4)` for year-over-year rates. The unemployment gap is
UNRATE minus NROU; tightness is JTSJOL divided by UNEMPLOY. Relative energy
inflation subtracts **core CPI** inflation from energy CPI inflation. Adaptive
expectations use only the preceding four quarters.

The regression pipeline uses OLS with Newey–West standard errors (four lags),
60-quarter rolling windows, a slope-break scan, and a kink search over
vacancy/unemployment ratios from 0.40 to 1.40. Thresholds in the holdout
comparison are selected from training observations only.

The estimated equations, with π for quarterly annualized inflation:

- Accelerationist, 1960–2026: headline π minus its mean over the previous
  four quarters = α + κ·(u − u\*), optionally plus γ·relative energy
  inflation. The slope-break test and the rolling windows use this form.
- Survey expectations, 1978–2026: core π minus Michigan 1-year expectations
  = α + κ·(u − u\*) + γ·relative energy inflation.
- Linear tightness, 2001Q1–2026Q1: core π minus Cleveland Fed 1-year
  expectations = α + β·(v/u) + δ·GSCPI.
- Kinked tightness, same sample: the same left side = α + β₋·min(v/u − c, 0) +
  β₊·max(v/u − c, 0) + δ·GSCPI, with c chosen by grid search. The slope
  ratio in the results is β₊/β₋.

A robustness grid re-estimates the kinked model for three inflation
measures, three expectations proxies and two supply controls.

Inference is exploratory. Both bootstraps resample eight-quarter blocks. The
linearity bootstrap (499 replications) uses seed 42 and the same 0.01-step threshold
grid as the observed statistic. Both tests use the add-one Monte Carlo
formula `(1 + exceedances) / (B + 1)`, so finite simulations never report zero.
The stored `c_ci95` bounds are an approximate LR search set: the borrowed
threshold cutoff has not been calibrated for this continuous-kink model.
Because the kinked model nests the linear one, the largest normalized SSR distance on the profile is bounded by about
5.10, below the 7.35 cutoff, so the set necessarily covers the whole grid.
The search profile uses `n * (SSR(c) - SSR_min) / SSR_min`. The slope-break p-value comes from a
fixed-regressor moving-block bootstrap of the no-break model (1,999
replications over every trimmed break date, with its own seed) under a stable no-break null. Residual blocks preserve local serial
dependence but do not guarantee validity under arbitrary intercept or
conditional-variance breaks.
The kink AIC counts the searched threshold (253.8 versus 254.8 for the linear
model); its conditional value before that penalty is retained as 251.8. This
is a selection heuristic for a nonregular model. Reported slope standard
errors condition on the selected threshold. The
results do not establish a nonlinear Phillips curve or causal effects.

## Files

- `code/`: data acquisition, transformations, estimation, figures, tables,
  and the synthetic demo (`run_demo.py`).
- `results/`: `results.json` holds every estimate and test, keyed by
  section (`descriptives`, `panelA_accelerationist`, `panelA_supply`,
  `panelC_survey`, `rolling_slope`, `quandt_andrews`, `modern_linear`,
  `modern_ugap`, `modern_kink`, `kink_test`, `decomposition_summary`,
  `oos_pre2020`, `oos_pre2023`, `robustness`); unavailable statistics are
  JSON `null`. `rolling_slope.csv` repeats the rolling slopes,
  `kink_grid_ssr.csv` gives the residual sum of squares at each threshold,
  and `decomposition.csv` gives the quarterly contributions behind Figure 6.
- `figures/`: seven research figures.
- `tests/`: offline checks using synthetic inputs and saved outputs.

Raw inputs and the processed dataset are not bundled; the demo generates
synthetic stand-ins locally. `decomposition.csv` includes 26 quarters of the
processed v/u, GSCPI and Cleveland Fed expectations series. Generated
Markdown tables are saved to the ignored `tables/` folder.

## Validation update: 2026-10-29

The audit aligned bootstrap grids, corrected the threshold-profile statistic
and AIC count, added strict source and calendar validation, repaired small
holdout prediction cases, and hardened demo output isolation. New regression
tests cover these failures with independent numerical checks. Coefficients,
standard errors, holdout projections and decomposition values remain unchanged.
The original input vintage is preserved; fresh PCE and GSCPI downloads contain
historical revisions and are not substituted into the saved estimates.
