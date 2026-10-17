# Forecasting Pipeline: Design Decisions

Decisions made when wiring the forecasting models (LightGBM expected returns,
GARCH volatility) into the Mean-CVaR optimizer, and why. Sections 1 to 4
apply to `scripts/run_forecast_optimization.py`; section 5 covers the
forecast mode of the Streamlit rebalancing app.

## 1. Risk aversion (λ) is scaled from historical data, not from the forecasts

### The objective

With no CVaR cap set, the optimizer chooses stock weights `w` to

```
minimize   λ_used · CVaR95(w)  −  Σ_i w_i · μ_i
```

- `μ_i` is stock i's expected daily return (historical average, or the forecast).
- `CVaR95(w)` is the average loss over the worst 5% of the 10,000 simulated
  daily scenarios, as a fraction of portfolio value.
- `λ_used` sets how much expected return one unit of CVaR must pay for.

### Why λ is rescaled at all

Daily expected returns (about 0.05%) are 30–60 times smaller than daily CVaR
(about 1.5–3%). Used directly, λ = 1 would mean "1% more CVaR must earn 1%
more return per day", which no stock does. The optimizer would ignore returns
and just minimize CVaR.

The package therefore rescales λ by default (`scale_risk_aversion=True` in
`ApiSettings`, `src/settings.py`):

```
λ_used = λ_set × max over the 30 stocks of ( μ_i / CVaR_i )
```

where `CVaR_i` is the CVaR of a position in stock i alone. This puts λ on a
usable scale: λ = 1 means risk costs the return-per-unit-of-CVaR of the best
single stock; above 1 is more cautious, below 1 more aggressive.

### The problem when the inputs are forecasts

The package computes that ratio from whatever `μ` and scenarios it is given.
When those are forecasts, `λ_used` moves with the forecasts and cancels
their overall level:

- **Returns:** if every forecast return is halved, `λ_used` halves too, the
  whole objective halves, and the optimal weights are unchanged. A
  uniformly gloomier outlook does not make the portfolio more defensive.
- **Volatility:** if GARCH forecasts higher volatility for every stock, each
  `CVaR_i` rises, `λ_used` falls by about the same factor, and the extra risk
  is largely cancelled — the opposite of what the volatility forecast is for.

Measured on 30 stocks, 10,000 scenarios, prices to 2026-09-25, halving every
forecast return:

| λ scaling | Change in weights (sum of absolute differences) | Portfolio CVaR |
|---|---|---|
| from forecasts (package default) | 1.2% (solver noise) | 1.422% → 1.423% |
| from historical data | 10.9% | 1.396% → 1.388% (more defensive) |

### Decision

Compute the ratio once from **historical** returns and scenarios, then hold
it fixed for the forecast run. λ keeps its usable scale, and a change in the
level of the forecasts changes the portfolio.

### How to configure it

`scripts/run_forecast_optimization.py --lambda-scaling {historical,forecast,off}`

| Option | λ used in the objective |
|---|---|
| `historical` (default) | `--risk-aversion` × max_i(historical μ_i / CVaR_i) |
| `forecast` | `--risk-aversion` × max_i(forecast μ_i / CVaR_i) — the package's built-in behaviour |
| `off` | `--risk-aversion` as given (λ = 1 then heavily favours low CVaR) |

The script prints the λ actually used. λ plays no role when `--cvar-limit` is
set: that form maximizes expected return subject to CVaR ≤ the cap.

## 2. Scenarios are rescaled to the forecast mean and GARCH volatility

### What was happening before

The 10,000 scenarios come from a kernel density estimate (KDE) fitted on
historical daily log returns. The return forecast replaced `μ` in the
expected-return term, but the scenarios — and therefore the CVaR term — were
still purely historical, and the GARCH forecast was computed but unused.

### What happens now

After the KDE scenarios are generated and before the solve, each stock's
scenarios are shifted and stretched
(`portfolio_optimization.forecasting.scenario_adjustment.rescale_scenarios`):

```
r_new[i, s] = μ̂_i + σ̂_i × (r[i, s] − m_i) / s_i
```

- `μ̂_i`: LightGBM forecast daily return for stock i.
- `σ̂_i`: GARCH forecast daily volatility for stock i
  (`CovarianceForecaster.predict_volatility`).
- `m_i`, `s_i`: mean and standard deviation of stock i's own 10,000 scenarios,
  so the result has exactly the forecast mean and volatility.

What is kept from history: each stock's distribution shape (fat tails, skew)
and the correlation between stocks in each scenario. What comes from the
forecasts: each stock's centre and spread. Both terms of the objective now
use the same forecasts.

### How to configure it

`scripts/run_forecast_optimization.py --scenario-adjustment {forecast,none}`

| Option | Scenarios used for CVaR |
|---|---|
| `forecast` (default) | rescaled to forecast mean and GARCH volatility |
| `none` | historical KDE scenarios; only the expected-return term uses the forecast |

### Measured effect (prices to 2026-09-25, λ scaling `historical`, measured with the old KDE bandwidth of 0.01; see section 3)

| Scenarios | Portfolio CVaR | Positions ≥ 0.5% |
|---|---|---|
| historical KDE (`none`) | 1.40% | 19 |
| rescaled (`forecast`) | 1.09% | 15 |
| rescaled, GARCH volatility × 1.5 for every stock (test) | 1.65% | 16 |

Most of the drop from 1.40% to 1.09% CVaR is the rescaling removing the
old KDE's 17% volatility inflation (section 3), not GARCH: GARCH currently
forecasts volatility about equal to the 3-year average.

The × 1.5 test moved only 5% of the weights. That is expected: the script
keeps the portfolio fully invested (`c_max=0`), and raising every stock's
volatility by the same factor barely changes which stocks are relatively
riskier.

## 3. Scenarios come from a KDE fitted on whitened returns

### The problem with the package's KDE on daily returns

The package's KDE (`fit_type="kde"`) smooths by adding random noise of a
fixed size, `bandwidth`, to each stock independently. The package default,
used in all its notebooks and the Streamlit demo, is 0.01 — i.e. 1% daily
noise, comparable to the stocks' own daily volatility (1–5%). The result
does not match history:

| Scenario generator | Volatility vs history (median stock) | Avg correlation (history 0.187) | Stock 95% CVaR vs history |
|---|---|---|---|
| package KDE, bandwidth 0.01 (package default) | 1.17× | 0.137 | 1.12× |
| package KDE, bandwidth 0.005 | 1.05× | 0.171 | 1.04× |
| package KDE, bandwidth 0.0025 | 1.01× | 0.183 | 1.02× |
| package KDE, bandwidth 0.001 | 1.00× | 0.187 | 1.01× |
| whitened KDE, bandwidth 0.3 | 1.04× | 0.187 | 1.04× |

(30 stocks, 753 trading days to 2026-09-25, 10,000 scenarios.)

Too much volatility overstates each stock's risk; too little correlation
understates how much stocks fall together, so portfolio CVaR is understated.
Any user of the package on daily returns with the default gets this.

Shrinking the bandwidth fixes both, but a very small bandwidth just
resamples the historical days (10,000 draws from 753 days ≈ each day 13
times) and no longer generates new scenarios. And the right fixed value
depends on the stocks' volatility and the return period (daily, weekly).

### Decision

Fit the KDE on **whitened** returns (`KDESettings(whiten=True)`, implemented
in `cvar_utils.generate_samples_whitened_kde`): transform the history to
uncorrelated, unit-volatility values, run the KDE there, and transform the
samples back. The transform uses the eigen-decomposition of the covariance,
so it also works when there are fewer trading days than stocks (e.g. the
S&P 500 over the 2007–2009 crisis). The smoothing noise then follows the
stocks' real correlation instead of being independent, so:

- correlation is preserved at any bandwidth;
- every stock's volatility is inflated by the same small factor
  (√(1 + bandwidth²) = 1.04× at 0.3), which the GARCH rescaling (section 2)
  removes;
- the bandwidth is in standard deviations, so one value works for calm and
  volatile stocks and for any return period.

With correlation restored, the forecast portfolio's CVaR is 1.22% (versus
1.09% with the old bandwidth): the old setting was understating risk.

### How to configure it

It is a setting on the package's KDE configuration, so it works wherever
scenarios are generated:

```python
KDESettings(bandwidth=0.3, kernel="gaussian", device="GPU", whiten=True)   # whitened
KDESettings(bandwidth=0.01, kernel="gaussian", device="GPU")               # package original
```

The library default stays `whiten=False` (unchanged from NVIDIA's release).
Where it is switched on in this repo:

| Where | How to choose |
|---|---|
| `scripts/run_forecast_optimization.py` | `--scenario-generator {whitened-kde,kde} --kde-bandwidth X` |
| Streamlit app (`demo/rebalancing_streamlit_app.py`) | sidebar: "Scenario Generator" and "KDE Bandwidth" |
| Notebooks `cvar_basic`, `efficient_frontier`, `rebalancing_strategies`, `launchable` | the `KDESettings(...)` line in each scenario-settings cell (commented with the original setting) |

| Option | Default bandwidth | Bandwidth units |
|---|---|---|
| `whitened-kde` (default) | 0.3 | standard deviations |
| `kde` (package KDE) | 0.0025 (not the package's 0.01) | daily return |

The script prints a fidelity line for every run — scenario volatility and
average correlation versus history — so a poor setting is visible.

## 4. Known limitations

### Everything runs on a daily scale

The return forecast horizon is 1 trading day (`FORECAST_HORIZON`, default 1),
the GARCH covariance is daily, and the CVaR scenarios are built from daily
returns. Changing the horizon requires rescaling the covariance and the
scenarios to the same horizon; otherwise the return term and the risk term of
the objective are on different scales.

### The return forecast has little measured skill

Measured out of sample on `dow30.csv` (trained to 2022-01-10, evaluated from
2022-01-11 to 2024-12-31), the LightGBM return forecast is indistinguishable
from predicting the training-set mean for every stock: about 0.02% lower error
at a 1-day horizon, and 52% directional accuracy (50% is a coin flip). Its
predictions vary only about 3% as much as realized returns. The pipeline is a
working template for plugging a return forecast into the optimizer; a
production deployment should supply a stronger return model.

## 5. Forecast mode in the rebalancing app

The Streamlit app (`demo/rebalancing_streamlit_app.py`) replays a past
period and re-optimizes along the way. By default it uses historical
averages, NVIDIA's original behaviour. A toggle switches on forecast mode,
which feeds the bundled LightGBM and GARCH models into every optimization.
The bundled models are placeholders (section 4): the point is the pattern, a
customer's own model plugged into the GPU optimizer, not the forecasts.

### No look-ahead

The app is a backtest, so a forecast used on a past date may only see data
up to that date. The deployed model endpoint is trained on data through
today and is therefore not used here; it fits only a portfolio for today
(`scripts/run_forecast_optimization.py`). The app trains its own copies of
the models.

### Decision: train at the start, retrain every 3 months

- The models are first trained on data strictly before the backtest's start
  date, then retrained every 3 months of the backtest, each time on data up
  to that retraining date only.
- The retraining dates follow a fixed calendar, not the rebalance dates. The
  GPU and CPU sides rebalance on different dates (each draws its own random
  scenarios), but a calendar is the same for both, so all training runs once,
  before the GPU-versus-CPU comparison starts, and both sides share the
  models. Training time is shown as its own step and does not count against
  either side's solve time.
- At each rebalance, the latest model trained on or before that date makes
  the forecast: LightGBM computes features from prices up to that date, and
  GARCH keeps its fitted parameters and updates its volatility forecast with
  the returns up to that date (no refit).
- Retraining at every rebalance was rejected: measured on 2 CPUs it costs
  10 to 25 seconds per rebalance for 30 stocks, which would bury the GPU's
  solve time of about 2 seconds for the whole run.

The forecast is applied as in the script: the expected-return vector is
replaced by the LightGBM forecast, scenarios are rescaled to the forecast
mean and GARCH volatility (section 2), and risk aversion is scaled from
historical data (section 1).

### Cost grows faster than the number of stocks

One LightGBM input feature is the average correlation of each stock with
every other stock, so its cost grows roughly with the square of the number
of stocks. Measured for one training plus forecast (2 CPUs):

| Stocks | Time | Peak memory |
| --- | --- | --- |
| 30 | 9 s | 0.3 GB |
| 100 | 19 s | 0.4 GB |
| 200 | 45 s | 0.9 GB |

At 380 stocks (`sp500`) it is several times slower and needs several GB of
memory; it exceeded a 4 GB session. The app shows this warning when forecast
mode is used with a large dataset. Forecast mode is intended for `dow30`.

