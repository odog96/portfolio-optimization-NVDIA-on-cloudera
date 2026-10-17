# Changelog

All notable changes to this project are documented here, one PR per entry,
newest first, dated by merge to `main`. Backfilled from git history.

## 2026-10-17 — branch `cai-forecasting`

Scenario fidelity fix for the package KDE, and forecasts wired into both
terms of the Mean-CVaR objective. Rationale and measurements:
`docs/forecasting_design_decisions.md`.

**Core package (candidates to propose upstream to NVIDIA)**

- Added `KDESettings.whiten` (default `False`, so existing behaviour is
  unchanged). When `True`, `generate_cvar_data` fits the KDE on whitened
  (decorrelated, unit-variance) returns via the new
  `cvar_utils.generate_samples_whitened_kde` and maps the samples back, and
  `bandwidth` is in standard deviations. Works when there are fewer dates than
  assets.
- Why: the standard KDE adds independent noise of size `bandwidth` to each
  asset. At the default 0.01 on daily returns (30 Dow stocks, 3 years) the
  scenarios are ~17% more volatile than history and the average pairwise
  correlation falls from 0.187 to 0.137, understating how much stocks fall
  together. Whitened KDE at 0.3: volatility 1.04× (uniform across assets),
  correlation 0.187.
- Notebooks (`cvar_basic`, `efficient_frontier`, `rebalancing_strategies`,
  `launchable`) now use `KDESettings(bandwidth=0.3, ..., whiten=True)`, with
  the original setting in a comment and a markdown note in the scenario
  generation sections.
- Streamlit app: sidebar "Scenario Generator" (Whitened KDE / Standard KDE)
  and "KDE Bandwidth" controls; defaults Whitened KDE, 0.3.
- Tests: `tests/test_core.py` — whitened KDE keeps correlation and inflates
  volatility uniformly (settings as model and as dict); works with fewer
  dates than assets.
- Not changed: the backtester's own KDE simulation
  (`portfolio_backtester._generate_samples_kde`, bandwidth 0.005, standard
  KDE).

**Forecasting layer (this fork)**

- `scripts/run_forecast_optimization.py`:
  - `--lambda-scaling {historical,forecast,off}` (default `historical`). The
    package rescales λ by max over stocks of (mean return / CVaR), computed
    from whatever returns it is given; computed from forecasts this cancels
    the forecasts' overall level (halving every forecast return left the
    weights unchanged). `historical` computes the ratio from historical data
    and holds it fixed. The λ actually used is printed.
  - `--scenario-adjustment {forecast,none}` (default `forecast`): rescale each
    stock's scenarios to the LightGBM forecast return and GARCH forecast
    volatility, so the CVaR term uses the forecasts. Previously the GARCH
    forecast was computed but never reached the optimizer.
  - `--scenario-generator {whitened-kde,kde}` and `--kde-bandwidth` (defaults
    whitened KDE, 0.3; standard KDE default lowered to 0.0025).
  - Prints scenario fidelity (volatility and correlation versus history).
- `forecasting/scenario_adjustment.py`: new `rescale_scenarios`.
- `forecasting/garch_model.py`: new `CovarianceForecaster.predict_volatility`
  (per-stock volatility vector; `predict` now builds on it, output unchanged).
- Tests: new `tests/test_forecasting.py` for `rescale_scenarios`.
- Docs: new `docs/forecasting_design_decisions.md`; `CLAUDE.md` updated.

**Cloudera Blueprint format**

- New top-level `README.md` in the Cloudera Blueprint template format
  (https://github.com/kevinbtalbert/Cloudera-Blueprints-Standard) and new
  `METADATA.yaml` catalog file. Fields still to fill in are marked TODO
  (Reprise demo link, final public GitHub link, published date, target
  vertical confirmation).
- NVIDIA's original README moved unchanged to `docs/NVIDIA_README.md`
  (relative links adjusted); links to it from `src/readme.md`,
  `notebooks/readme.md` and `CLAUDE.md` updated.
- New `assets/cloudera_architecture.svg` (and PNG export): architecture of
  the blueprint on Cloudera AI.

**Cleanup**

- Streamlit demo: the GPU solver (cuOpt) is pinned to the PDLP method, as
  everywhere else in the repo. Its default (concurrent) method segfaulted
  intermittently during re-optimization and took the whole app down. The
  GPU side now runs in its own process like the CPU side, and a solver
  process that dies is reported as an error instead of hanging the page.
- Deployment scripts (`scripts/install_dependencies.py`, `download_data.py`,
  `train_and_register.py`, `deploy_model.py`, `launch_streamlit.py`) no longer
  fail with `NameError: __file__` when Cloudera AI runs them in its
  cell-by-cell kernel; they fall back to the working directory (the project
  root).
- `scripts/launch_streamlit.py` starts Streamlit headless (it otherwise waits
  for a first-run email prompt and the Application never leaves Starting),
  via the current Python interpreter, from the project root.
  It also reinstalls pydantic so its dependencies exist in the
  Application's runtime (all Python 3.12 runtimes share `~/.local`).
- Removed temporary deployment experiments: `hello.py` (model-build smoke
  test), `serve.py` and `predict.py` (hand-written serving entry points, not
  used: `scripts/deploy_model.py` serves the registered MLflow model) and
  `cml_build.sh` (their build step).
- `INVESTIGATION_NOTES.md` removed from the repository (kept locally, gitignored);
  known limitations moved to `docs/forecasting_design_decisions.md` section 4.
- `.gitignore`: workspace home-directory files (the project root is the
  Cloudera AI user's home directory) and the personal `busy_timer.ipynb`.

**Forecast mode in the rebalancing app**

- `rebalance_portfolio` takes an optional `estimate_provider` hook, called
  before each optimization with the scenario data, CVaR parameters, prices
  up to the rebalance date only, and that date. Default None: NVIDIA's
  behaviour is unchanged (candidate for upstream).
- `forecasting/estimate_provider.py`: `ForecastEstimateProvider` trains
  LightGBM and GARCH before the backtest start and every 3 months, each on
  earlier data only, and applies the forecasts at each rebalance (forecast
  mean, rescaled scenarios, risk aversion scaled from historical data).
- Streamlit app: "Use forecasts" toggle, off by default. Models are trained
  once before the GPU-vs-CPU race and shared by both sides; forecast time is
  reported separately. Warning on datasets over 50 stocks (cost grows faster
  than the number of stocks).
- Fix: `CovarianceForecaster.predict_volatility` now conditions on the prices
  passed in; it used to forecast from the end of the training window
  regardless of its input.
- Design: `docs/forecasting_design_decisions.md` section 5.

**AMP install and demo operations**

- Streamlit Application stops itself after 60 minutes without user
  interaction (`IDLE_SHUTDOWN_MINUTES`, `0` disables), so an abandoned demo
  no longer holds the workspace GPU. The app touches a heartbeat file on
  every rerun; `scripts/launch_streamlit.py` watches it and stops the
  Application. Verified in a running Application on 2026-10-15.
- Results page notes that each side draws its own random scenarios, so the
  GPU and CPU portfolios differ; the comparison is speed, not returns.
- AMP step 2 (Install GPU Packages) no longer requests a GPU: the pip install
  and import check run on CPU.
- Validate GPU Pipeline is now a job the AMP creates but does not run; users
  run it from the Jobs page before starting work. The application is the
  only AMP step that requests a GPU. The check solves through CVXPY with
  cuOpt's PDLP method pinned, the same path as the Streamlit app.
- Step 1 (`scripts/install_dependencies.py`) no longer installs pandas 3 only
  to replace it with pandas 2.3.
- Model training and registration is now an optional job (Train and Register
  Forecast Models) that the AMP creates but does not run, as the demo
  application trains its own models.
- Model deployment is now an optional job (Deploy Forecast Model) that the
  AMP creates but does not run; the demo application does not use the
  endpoint. As a job it waits up to 25 minutes for the model build and fails
  clearly if the build does not finish.

## 2026-07-17 — PR #56

Guard against None CVXPY variable values when the solver hits a resource
limit before finding a feasible incumbent.

- Added a check in `_solve_cvxpy_problem` that raises a `RuntimeError`
  immediately after `solve()` if `self.w.value is None`, before any
  downstream `.value` access occurs.
- Follows from `cvxpy` PR #3443, which downgrades cuOpt's status to
  `INFEASIBLE_INACCURATE` when a time limit is hit with no incumbent
  found. Variable `.value`s are now `None` in that case instead of
  returning an empty primal solution.
- Once the upstream `cvxpy` crash was fixed, this initially surfaced
  as a `TypeError` in `_get_cvxpy_risk_metric_value`; this new check
  catches it earlier and provides the `cvxpy` solver status for debugging.

## 2026-07-03 — PR #54

Clarify daily return units and add a GPU-vs-CPU composition comparison in
`notebooks/mean_variance_basic.ipynb`.

- Added a units note and per-statistic labels: all figures derive from 1-day
  LOG returns and are daily, not annualized; added an annualized block
  (×252 log return, exponential conversion to simple return, √252 scaling for
  volatility/Sharpe).
- Added a portfolio-composition comparison cell (direct cuOpt vs CVXPY-cuOpt
  vs CLARABEL weights side by side with per-weight differences), complementing
  the metrics-only `compare_results`.
- Renamed the CVXPY-API solve results to `cvxpy_gpu_*` so the direct cuOpt
  solve no longer shadows them.
- Notebook re-executed end-to-end on GPU (cuOpt 26.06, fresh S&P 500 snapshot).

## 2026-06-26 — PR #52

Reconcile the project and package version strings.

- Set `pyproject` `[project].version` and `src/__init__.py` `version`
  (exported as `portfolio_optimization.version`) both to `26.4`, replacing the disagreeing
  `25.10` / `1.0.0` and aligning with the cuOpt 26.4 release line and the
  `cuopt-cu13==26.4.*` dependency-pin style.
- Regenerated `uv.lock`.

## 2026-06-26 — PR #50

Add cuOpt SOCP variance-cap support to the Mean-Variance optimizer.

- Added a direct cuOpt SOCP path for Mean-Variance variance-cap constraints:
  the variance cap is modeled as a convex quadratic (QCQP) row via
  `QuadraticExpression` that cuOpt converts to second-order-cone form and solves
  with the barrier method (`src/mean_variance_optimizer.py`,
  `src/mean_variance_parameters.py`).
- Added `_covariance_for_quadratic_terms` to sanitize the covariance for
  QP/SOCP terms (shape/finiteness checks, symmetrization, and PSD enforcement
  via a minimum-eigenvalue diagonal floor).
- Bumped cuOpt/cuML to 26.06 for the `cuda12` extra and added a cuOpt-only
  `cuda13-socp` extra (`cuopt-cu13==26.6.*`) for CUDA 13 SOCP solves, since
  `cuml-cu13` 26.06 is not yet published; `cuda13` still tracks 26.04.
- Switched cvxpy from the pinned git-master build to the released
  `cvxpy>=1.9.2` and dropped the `[tool.uv.sources]` git pin.
- Added SOCP unit + GPU tests (`tests/test_core.py`), a SOCP benchmark workflow
  (`tests/benchmarks/benchmark_workflows.py`, `tests/benchmarks/thresholds.toml`,
  `tests/test_skill_benchmarks.py`), and the `mean-variance-socp-var-limit`
  eval case (`skills/cufolio/evals/evals-full.json`, now 10 cases).
- Updated docs: `skills/cufolio/SKILL.md`,
  `skills/cufolio/references/workflows/agent_recipes.md`,
  `skills/cufolio/skill-card.md`, `skills/cufolio/evals/EVAL.md`, and the root
  `README.md` install instructions for the new extra.

## 2026-06-05 — PR #47

Add the public Streamlit rebalancing demo.

- Added `demo/rebalancing_streamlit_app.py`, shared demo parameters, demo
  requirements, Streamlit theme config, and supporting architecture / benchmark
  / animation media.
- Updated the root README and demo README with Docker port publishing,
  dataset-download, and remote Streamlit launch instructions.

## 2026-06-02 — PR #46

Add the portfolio optimization agent skill for the NVIDIA skills catalog.

- Added agent/plugin manifests, `skills/cufolio/SKILL.md`, skill card,
  benchmark docs, eval definitions, and workflow recipes.
- Added skill validation workflow/utilities plus benchmark and static skill tests.
- Hardened portfolio optimization APIs and docs around scenario inputs, portfolio behavior,
  and benchmark workflows.

## 2026-05-28 — PR #45

Fix efficient-frontier benchmark behavior for zero-risk portfolios.

- Treat zero-volatility efficient-frontier points as undefined Sharpe ratios
  instead of allowing `0/0` to participate in max-Sharpe selection.
- Added regression coverage in `tests/test_core.py`.

## 2026-05-27 — PR #41

Refresh source workflows for backtesting and rebalancing.

- Updated `src/backtest.py`, `src/rebalance.py`, and related CVaR utility logic.
- Added tests covering the refreshed workflow behavior.

## 2026-05-26 — PR #42

Enhance notebook setup and reduce CI/logging noise.

- Updated the main CI workflow and `notebooks/launchable.ipynb` setup flow.
- Reduced noisy data-download logging in `src/utils.py`.

## 2026-05-04 — PR #37

Add `CHANGELOG.md`, backfilled from `git log --merges --first-parent origin/main`.
One entry per merged PR, newest first, `## YYYY-MM-DD — PR #N` headers.
No source/config changes.

## 2026-04-22 — PR #40

Fix invalid notebook JSON to satisfy the `nbformat` schema.

- Regenerated / repaired notebook JSON for launchable, efficient-frontier,
  and mean-variance notebooks.

## 2026-04-22 — PR #39

Sync launchable notebooks with source updates and add Mean-Variance coverage.

- Updated Brev and launchable notebook content/setup.
- Added related dependency, license, utility, and test updates.

## 2026-04-22 — PR #38

Update launchable notebook API usage for cuOpt 26.4 and the Pydantic settings
migration.

## 2026-04-22 — PR #36

Fix notebook cuOpt presolve settings for cuOpt 26.4 compatibility.

- Changed notebook presolve values from `False` to `0` where required.

## 2026-04-21 — PR #34

**Breaking — minimum Python bumped from `>=3.10` to `>=3.11`.** Required by
`cuml-cu13==26.4`. Also updates `target-version` in the black config and
`CONTRIBUTING.md`.

## 2026-04-20 — PR #33

Switch `cvar_basic.ipynb` to render the architecture diagram via PNG to
avoid SVG rendering issues in `nbconvert` output.

## 2026-04-20 — PR #32

Large bundled change:

- **Added** — Mean-Variance (Markowitz) optimizer with CVXPY and cuOpt
  backends; `notebooks/mean_variance_basic.ipynb`.
- **Added** — `BaseOptimizer` / `BaseParameters` classes and Pydantic
  settings models (`ReturnsComputeSettings`, `ScenarioGenerationSettings`,
  `KDESettings`, `ApiSettings`) under `portfolio_optimization.settings`.
- **Added** — SP100, DOW30, and Global Titans dataset download support in
  `utils.download_data`.
- **Added** — `compute_absolute_returns` (simple-diff semantics) as a
  distinct function.
- **Breaking — API rename** — `compute_abs_returns` → `compute_linear_returns`
  (pct-change). Previous `diff` semantics now live in `compute_absolute_returns`.
- **Breaking — settings API** — `calculate_returns`, `generate_cvar_data`,
  and optimizer constructors now take Pydantic settings models instead of
  plain dicts.
- **Breaking — dependency pins** — `cuml-cu{12,13}` and `cuopt-cu{12,13}`
  bumped from `25.12.*` to `26.4.*` / `26.04.*`; `pydantic>=2.12.5` added.
- **Changed** — `optimize_market_regimes()` unified to support both CVaR
  and mean-variance.
- **Fixed** — cuML KDE sampler returns numpy arrays via `using_output_type`.
- **Fixed** — `quad_form` numerical issues resolved via `psd_wrap` on the
  covariance matrix.
- Notebooks rewritten against the new API; outputs regenerated.

## 2025-12-17 — PR from fork (phuo-nv)

- **Changed** — cuOpt model construction switched to `LinearExpression` to
  avoid recursion depth errors on large problems.
- **Changed** — refactored returns-calculation methods; made exception
  handling explicit.
- **Added** — KDE timing instrumentation in the CVaR pipeline.
- **Fixed** — lazy imports in `cvar_optimizer` to avoid eager GPU-module
  load at import time.

## 2025-12-02 — PR #28

Small touch-up of `notebooks/launchable.ipynb` (1 file, +161/-30).

## 2025-12-02 — PR #27

Minor launchable-notebook fix (4 files, +33/-10).

## 2025-12-02 — PR #26

Major launchable-notebook iteration: GPU checks, setup steps, regenerated
outputs (7 files, +3095/-173).

## 2025-12-01 — PR #25

Rename / replace the architecture diagram asset (`docs/arch_diagram.png`).

## 2025-12-01 — PR #24

Refresh notebook outputs to match latest src behavior (4 files, +213/-171).

## 2025-11-26 — PR #23

Final pass on the stock-data directory setup (4 files, +1596/-1813) —
regenerated dataset artifacts and README polish.

## 2025-11-25 — PR #22

Add a bundled dataset file (1 file, +432/-12).

## 2025-11-25 — PR #21

Polish on the stock-data directory layout (3 files, +15/-2).

## 2025-11-25 — PR #20

Initial `data/stock_data/` scaffolding (5 files, +51/-68).

## 2025-11-25 — PR #19

Small source touch-up (1 file, +7/-6).

## 2025-11-25 — PR #18

Launchable-notebook refinements (7 files, +1187/-1326).

## 2025-11-24 — PR #17

Add Brev-deployable `notebooks/launchable.ipynb` (9 files, +4855/-747).

## 2025-11-24 — PR #16

Finalize `uv.lock` migration and sequence diagram (5 files, +1377/-1091).

## 2025-11-23 — PR #15

Refine `uv.lock` sequence diagram docs (6 files, +67/-49).

## 2025-11-21 — PR #14

Commit `uv.lock` and migrate project to `uv` as the dependency manager;
add a sequence diagram documenting the dependency-management flow
(5 files, +2928/-6).

## 2025-11-21 — PR #13

Add `release-2512.yml` workflow, `docs/arch_diagram.png`, large notebook
regeneration, and `src/` cleanup (11 files, +1505/-2082).

## 2025-11-20 — PR #12

Small notebook fixes in `cvar_basic` and `rebalancing_strategies`
(2 files, +4/-4).

## 2025-11-19 — PR #11

Notebook regeneration and src cleanup in backtest / rebalance / utils
(5 files, +703/-912).

## 2025-11-19 — PR #10

Notebook and src refinements across cvar_basic, backtest, cvar_utils,
portfolio, rebalance, utils (7 files, +304/-141).

## 2025-11-17 — PR #9

README additions (+36 lines).

## 2025-11-13 — PR #8

`src/readme.md` fix (1 file, +2/-2).

## 2025-11-12 — PR #7

Remove `deploy/1_Deploy_Template.ipynb`; major `cvar_basic.ipynb` update;
add `notebooks/readme.md`; minor `src/cvar_utils.py` and `src/utils.py`
tweaks (7 files, +1358/-1235).

## 2025-11-12 — PR #6

README polish (1 file, +18/-12).

## 2025-11-12 — PR #5

Remove legacy scaffolding: `Maintainers/`, old `CHANGELOG.md`,
`README_ARCHIVED.md`, empty `customize/` and `evaluate/` readmes; README
polish (10 files, +74/-409).

## 2025-11-07 — PR #4

`LICENSE-3rd-party.txt` edits, remove a `pyproject.toml` entry
(2 files, +5/-7).

## 2025-11-06 — PR #3

Add `LICENSE-3rd-party.txt`; README touch-up (2 files, +119/-2).

## 2025-11-06 — PR #2

README polish (1 file, +2).

## 2025-11-06 — PR #1

**Initial code drop.** Core `portfolio_optimization` package: CVaR optimizer
(`portfolio_optimization.cvar_optimizer.CVaR`), portfolio backtester
(`portfolio_optimization.backtest.portfolio_backtester`), utils, scenario generation
(KDE / Gaussian / historical). Example notebooks: `cvar_basic.ipynb`,
`efficient_frontier.ipynb`, `rebalancing_strategies.ipynb`. `pyproject.toml`
with `cuda12` / `cuda13` extras pinning `cuml-cu{12,13}` and
`cuopt-cu{12,13}`. `README.md`, `CONTRIBUTING.md`, `CODEOWNERS`,
NVIDIA-AI-Blueprints branding and issue templates (22 files, +10885/-25).
