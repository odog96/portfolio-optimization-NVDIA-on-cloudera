# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- AI agents: also read AGENTS.md and skills/portfolio-optimization/SKILL.md before using this package's workflows. -->

## What this is

GPU-accelerated portfolio optimization (Mean-CVaR and Mean-Variance/SOCP) built on
**NVIDIA cuOpt** (solver) and **RAPIDS cuML** (GPU KDE scenario generation). The
importable package is `portfolio_optimization`, whose code lives under `src/`
(see `[tool.setuptools.package-dir]` in `pyproject.toml` — `src/` is installed
*as* `portfolio_optimization`, it is not a `src/portfolio_optimization/` layout).

## Commands

Dependency management is via `uv`. Extras are mutually exclusive GPU/runtime
stacks; groups are dev-only tooling not published in package metadata.

```bash
# Install (pick one CUDA extra)
uv sync --extra cuda13 --group notebooks         # CUDA 13, full cuOpt/cuML 26.04 stack
uv sync --extra cuda12 --group notebooks         # CUDA 12, full cuOpt/cuML 26.06 stack
uv sync --extra cuda13-socp --group notebooks    # CUDA 13, cuOpt-only 26.06 (SOCP preview, no cuML)
uv sync --extra cuda13 --group dev               # add ruff/pytest/pre-commit

# Lint / format
uv run ruff format src/
uv run ruff check src/

# Tests
uv run pytest                          # CPU suite (GPU-marked tests self-skip without hardware)
uv run pytest tests/test_core.py       # single file
uv run pytest tests/test_core.py::test_name   # single test
uv run pytest -m gpu -rs               # GPU-only suite (needs cuOpt/cuML; -rs shows skip reasons)

# Pre-commit (formatting, ruff, shellcheck, yamllint, zizmor, skill-version validation)
uv run pre-commit run --all-files
```

There is no CPU solver fallback anywhere in this codebase — see "cuOpt-only" below.

## Versioning

`VERSION` at the repo root is the single source of truth. Never hand-edit a
version elsewhere. After changing `VERSION`, run:

```bash
./ci/utils/sync_skills_version.sh
```

This propagates it to `pyproject.toml`, `src/__init__.py`, `skills/*/SKILL.md`
frontmatter, `.claude-plugin/marketplace.json`, `.cursor-plugin/plugin.json`,
and `gemini-extension.json`. `ci/utils/validate_skills.sh` fails the build on
drift and both scripts are wired as pre-commit hooks. Editing `skills/*/SKILL.md`
invalidates its NVSkills signature (`skill.oms.sig`) — a maintainer must re-run
`/nvskills-ci` on the PR to reattach it.

## Architecture

All core modules live flat under `src/` (see `src/readme.md` for the fuller
module-by-module doc). The pipeline is: prices → returns → (scenarios) → optimize → portfolio → backtest/rebalance.

- **`base_parameters.py` / `base_optimizer.py`** — shared Pydantic parameter
  base and `BaseOptimizer` abstract base (weight-constraint handling, shared
  solve/state machinery) that both optimizers below extend.
- **`cvar_parameters.py` + `cvar_optimizer.py` (`CVaR`)** — Mean-CVaR optimizer.
  Solves via CVXPY (`cp.CUOPT`, `solver_method="PDLP"`) or the direct cuOpt
  Python API. Needs scenario data (see `cvar_data.py`, `cvar_utils.py`).
- **`mean_variance_parameters.py` + `mean_variance_optimizer.py` (`MeanVariance`)**
  — Mean-Variance/Markowitz optimizer with a hard variance cap (`var_limit`),
  solved as SOCP/QCQP directly through the cuOpt Python API
  (`ApiSettings(api="cuopt_python")`). Requires cuOpt with QCQP/SOCP support
  (26.06+); older builds (e.g. the `cuda13` extra's cuOpt 26.04) raise
  "Quadratic constraints not supported" — tests gate on this via the
  `require_cuopt_socp` fixture in `tests/conftest.py`.
- **`scenario_generation.py` (`ForwardPathSimulator`)** — synthetic scenario
  generation via GBM, independent of the KDE path used for CVaR.
- **`portfolio.py` (`Portfolio`)** — holds tickers/weights/cash for one
  allocation; the common currency passed between optimizers, backtester, and
  rebalancer.
- **`backtest.py` (`portfolio_backtester`)** — evaluates a `Portfolio` against
  benchmarks (historical, KDE-simulated, or Gaussian-simulated), producing
  Sharpe/Sortino/max-drawdown/cumulative-return metrics.
- **`rebalance.py` (`rebalance_portfolio`)** — rolling re-optimization over a
  price CSV on a schedule or drift trigger; wraps `cvar_optimizer` internally.
- **`settings.py`** — Pydantic settings models threaded through the pipeline:
  `ReturnsComputeSettings`, `ScenarioGenerationSettings`, `KDESettings`,
  `ApiSettings`. These are the primary way behavior is configured, not kwargs.
- **`utils.py` / `cvar_utils.py`** — data loading (`get_input_data`, multi-
  format), returns computation (`calculate_returns`, log/linear), plus (in
  `cvar_utils`) CVaR math, efficient-frontier construction
  (`create_efficient_frontier`), and solver benchmarking helpers.
- **`forecasting/`** — optional ML forecasting layer, imported separately as
  `portfolio_optimization.forecasting` (not re-exported from the top-level
  package `__init__.py`). See "Forecasting module" below.

Key shape gotcha: `returns_dict` (the object threaded through nearly every
function) is a flat dict with keys like `returns`, `mean`, `covariance`,
`tickers` — not nested/keyed by regime. `regime_dict` filters by date range
(`{"name": ..., "range": (start, end)}`), it does not carry tickers.
`solve_optimization_problem(...)` returns `(result_row, portfolio)`, never a
nested results dict.

### cuOpt-only, deliberately

This repo has a hard rule, enforced in `AGENTS.md`, `skills/.../SKILL.md`, and
CI: **always solve on the cuOpt GPU solver, never fall back to a CPU solver**
(no CLARABEL/SCS/ECOS/etc.). If cuOpt is unavailable, code/tests/agent
workflows should report that the GPU runtime is missing rather than silently
substituting a CPU solve.

### Forecasting module

`src/forecasting/` (package `portfolio_optimization.forecasting`) is a
newer, separate ML layer that replaces historical mean/covariance estimates
with model-based forecasts before they reach an optimizer. It is not covered
by `skills/portfolio-optimization/SKILL.md` or `AGENTS.md` yet, and has no
dedicated test file under `tests/`.

- **`config.py`** — Pydantic config tree: `ForecastingConfig` (top-level:
  `forecast_horizon`, `training_window`, `retrain_frequency`, `return_type`,
  optional `caii_endpoint` / `caii_covariance_endpoint`), plus nested
  `FeatureConfig`, `LightGBMConfig`, `GARCHConfig`.
- **`feature_engineering.py`** — `compute_features` turns a wide price
  DataFrame into per-asset technical features (momentum, rolling volatility,
  RSI, cross-asset correlation); `build_training_data` /
  `flatten_features_for_training` shape those into supervised-learning
  X/y for LightGBM.
- **`lightgbm_model.py` (`ReturnsForecaster`)** — single LightGBM model
  trained across all assets to predict forward mean returns.
- **`garch_model.py` (`CovarianceForecaster`)** — per-asset univariate GARCH
  (via `arch`) combined with a DCC-like correlation structure into a forward
  covariance matrix.
- **`caii_client.py` (`ForecastClient`)** — unified facade over the two
  models above. `predict_returns`/`predict_covariance`/`predict` run the
  local models unless `config.caii_endpoint` /
  `caii_covariance_endpoint` is set, in which case they call a deployed CAII
  (Cloudera AI Inference) endpoint via `OpenInferenceClient` instead
  (needs `CAII_API_KEY` in the environment). **`update_returns_dict(returns_dict,
  prices)` is the integration point back into the core pipeline** — it
  overwrites `returns_dict["mean"]`/`["covariance"]` in place of the
  historical estimates, so the result can be handed straight to
  `cvar_optimizer`/`mean_variance_optimizer` as normal.
- Worked example: `notebooks/forecasting_training.ipynb`.

### Cloudera AI Workbench deployment path

Alongside the README's Docker/uv workflow, `.project-metadata.yaml` defines a
separate Cloudera AI (CML) AMP packaging for this project, driven by scripts
under `scripts/`: `install_dependencies.py` / `install_gpu_packages.py` (env
setup), `download_data.py`, `train_and_register.py` (trains the LightGBM +
GARCH forecasters, logs them to MLflow, and registers in the CML model
registry), `validate_gpu.py` (end-to-end GPU smoke test), `deploy_model.py`
(reads `outputs/registration_info.json` from the train step and deploys the
registered model to a CML serving endpoint via `cmlapi`), and
`launch_streamlit.py` (starts the rebalancing demo as a CML application).
This path assumes a CML/CDSW runtime (`CDSW_PROJECT_ID` env var, `cmlapi`
installed) and is independent of the plain-Docker/uv install — don't assume
one implies the other. `models/` and `outputs/` (gitignored) hold the
artifacts these scripts produce locally.

### Agent skill

`skills/portfolio-optimization/SKILL.md` is the authoritative recipe for
driving this package as an agent workflow (defaults, constraint mapping,
validation rules, canonical code shapes); `references/workflows/agent_recipes.md`
has copyable full functions. Both `AGENTS.md` and `.agents/` (symlinks) point
here. Treat `SKILL.md` as source of truth over ad hoc reimplementation when
doing portfolio-optimization tasks through the package's public API.

## Testing structure

- `tests/test_core.py` — main CPU-runnable unit tests.
- `tests/test_skill.py` — validates the skill workflow shapes/routing.
- `tests/test_skill_benchmarks.py` + `tests/benchmarks/` — GPU-gated
  performance regression suite (`-m gpu`) that runs the actual SKILL.md
  workflows end-to-end and checks thresholds in
  `tests/benchmarks/thresholds.toml`; auto-skips without cuOpt/cuML.
- `tests/conftest.py` — shared fixtures, notably `require_cuopt_socp`, which
  skips SOCP/variance-cap tests unless the installed cuOpt is >= 26.6 (older
  builds lack QCQP support).
- CI (`ci/utils/run_gpu_tests.sh`) runs `pytest tests/ -v -m gpu -rs` inside a
  CUDA container per extra (`cuda13`, then `cuda13-socp` to cover SOCP since
  `cuda13`'s cuOpt 26.04 can't). `ci/utils/run_notebooks.sh` executes the
  example notebooks end-to-end via papermill as part of the same workflow.

## Notebooks & demo

- `notebooks/` — the canonical worked examples (`cvar_basic.ipynb`,
  `efficient_frontier.ipynb`, `rebalancing_strategies.ipynb`); CI runs these
  via papermill/nbconvert and checks for generated `*_result.html/.ipynb`.
  Requires the Jupyter kernel from `--group notebooks` (see README for
  `ipykernel install` step). `forecasting_training.ipynb` walks through the
  `forecasting/` module separately and is not mentioned in the README.
- `demo/` — standalone Streamlit dynamic-rebalancing app
  (`rebalancing_streamlit_app.py`); see `demo/README_streamlit.md`.
- Default dataset: `data/stock_data/sp500.csv` (gitignored, fetched on demand
  via `portfolio_optimization.utils.download_data`) — a historical snapshot
  that may not reflect current index constituents.
