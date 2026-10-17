"""End-to-end: market data -> forecasts -> scenarios -> Mean-CVaR portfolio.

This is the integrated flow the project exists to demonstrate, wired as a
single run:

  1. Pull prices (live through today by default, or a pinned date range).
  2. Expected returns from the deployed CML Model endpoint, or the local
     LightGBM model with --local.
  3. Volatility and covariance from the local GARCH model.
  4. Overwrite the historical mean/covariance with the forecasts.
  5. Generate KDE scenarios on GPU (cuML) from historical returns
     (--scenario-generator, default whitened KDE), then
     rescale each stock's scenarios to the forecast mean and GARCH
     volatility (--scenario-adjustment).
  6. Solve Mean-CVaR on GPU (cuOpt) and report the portfolio.

Requires a GPU session: cuOpt and cuML are mandatory. There is no CPU solver
fallback anywhere in this repo by design — if cuOpt is missing this script
fails loudly rather than silently substituting a CPU solve.

Horizon note: everything here runs on a DAILY scale. The forecast horizon is
1 trading day, the GARCH covariance is daily, and the scenarios come from
freq=1 daily returns. Changing the horizon requires rescaling the covariance
and the scenarios to match -- see docs/forecasting_design_decisions.md, section 4.

Usage
-----
    python scripts/run_forecast_optimization.py
    python scripts/run_forecast_optimization.py --local --num-scen 2000
    python scripts/run_forecast_optimization.py --start 2022-01-01 --end 2024-12-31
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_ROWS = 64   # rows needed for exactly one feature row per ticker
GARCH_ROWS = 500    # longer window so the per-ticker GARCH fits are stable
# KDE smoothing width per generator; see --kde-bandwidth.
DEFAULT_BANDWIDTH = {"whitened-kde": 0.3, "kde": 0.0025}
MODEL_NAME = os.environ.get("CAII_MODEL_NAME", "PortfolioReturnsForecaster")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", action="store_true",
                   help="Use the local LightGBM model instead of the deployed endpoint.")
    p.add_argument("--start", default=None, help="Price history start, YYYY-MM-DD.")
    p.add_argument("--end", default=None,
                   help="Price history end, YYYY-MM-DD. Default: latest trading day.")
    p.add_argument("--num-scen", type=int, default=10000, help="Scenarios to generate.")
    p.add_argument("--confidence", type=float, default=0.95, help="CVaR confidence level.")
    p.add_argument("--cvar-limit", type=float, default=None,
                   help="Hard CVaR cap as a fraction, e.g. 0.03. Omit for the "
                        "risk_aversion trade-off form.")
    p.add_argument("--risk-aversion", type=float, default=1.0,
                   help="Lambda in: minimize lambda * CVaR - expected return. "
                        "How it is scaled is set by --lambda-scaling.")
    p.add_argument("--lambda-scaling", choices=["historical", "forecast", "off"],
                   default="historical",
                   help="How --risk-aversion is rescaled before the solve. "
                        "historical (default): multiply by max over stocks of "
                        "(historical mean / CVaR), fixed so forecasts cannot move it. "
                        "forecast: same ratio but from the forecast returns (the "
                        "package's built-in default; a uniformly lower return or "
                        "higher volatility forecast then leaves the portfolio "
                        "unchanged). off: use --risk-aversion as given. Ignored "
                        "with --cvar-limit. See docs/forecasting_design_decisions.md.")
    p.add_argument("--w-max", type=float, default=0.20, help="Max weight per asset.")
    p.add_argument("--scenario-generator", choices=["whitened-kde", "kde"],
                   default="whitened-kde",
                   help="whitened-kde (default): KDE fitted on decorrelated, "
                        "unit-volatility returns, which keeps the historical "
                        "correlation between stocks. kde: the package's KDE, which "
                        "adds independent noise per stock and weakens correlation "
                        "as the bandwidth grows. See docs/forecasting_design_decisions.md.")
    p.add_argument("--kde-bandwidth", type=float, default=None,
                   help="KDE smoothing width. whitened-kde: in standard deviations "
                        "(default 0.3). kde: in daily return units (default 0.0025; "
                        "the package default 0.01 inflates volatility ~17%% and cuts "
                        "correlation ~25%% on daily returns).")
    p.add_argument("--scenario-adjustment", choices=["forecast", "none"],
                   default="forecast",
                   help="forecast (default): re-center each stock's scenarios on "
                        "its forecast return and rescale them to its GARCH "
                        "volatility, so the CVaR term uses the forecasts. none: "
                        "keep the historical KDE scenarios; only the expected-"
                        "return term uses the forecast. See "
                        "docs/forecasting_design_decisions.md.")
    p.add_argument("--historical-baseline", action="store_true",
                   help="Also solve with historical mean/covariance and compare.")
    return p.parse_args()


def require_gpu_stack():
    """Fail loudly if the GPU solver is unavailable. No CPU fallback."""
    import cvxpy as cp

    installed = {str(s) for s in cp.installed_solvers()}
    if not hasattr(cp, "CUOPT") or str(cp.CUOPT) not in installed:
        raise RuntimeError(
            "cuOpt GPU solver is required and was not found. Run this in a GPU "
            "session after scripts/install_gpu_packages.py. Do not substitute a "
            "CPU solver: it invalidates the benchmark this project exists to show."
        )
    return cp


def load_prices(tickers, start, end):
    """Pull closing prices for `tickers`, newest data by default."""
    import yfinance as yf

    if end is None:
        end_ts = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    else:
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    # Over-fetch on the calendar so we clear GARCH_ROWS *trading* days.
    start_ts = pd.Timestamp(start) if start else end_ts - pd.Timedelta(days=1100)

    raw = yf.download(tickers, start=start_ts.date(), end=end_ts.date(),
                      timeout=60, progress=False)["Close"]
    prices = raw.ffill().dropna(axis=1, how="any")
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        print(f"  WARNING: dropped for incomplete history: {missing}")
    return prices[[t for t in tickers if t in prices.columns]]


def predict_returns_endpoint(prices, feature_order, ticker_codes):
    """Score the last feature row per ticker against the deployed endpoint."""
    import cmlapi
    import requests
    from portfolio_optimization.forecasting.config import ForecastingConfig
    from portfolio_optimization.forecasting.feature_engineering import compute_features

    tickers = list(prices.columns)
    features = compute_features(prices.tail(FEATURE_ROWS), ForecastingConfig().features)
    if len(features) < 1:
        raise RuntimeError(
            f"No feature rows produced from {FEATURE_ROWS} price rows; "
            "need at least 64 trading days per ticker."
        )

    rows = []
    for t in tickers:
        last = features[t].iloc[-1]
        rows.append([float(last[c]) for c in feature_order[:-1]] + [float(ticker_codes[t])])
    payload = np.asarray(rows, dtype=np.float64)
    if np.isnan(payload).any():
        raise RuntimeError("NaNs in the feature payload; check the price history.")

    client = cmlapi.default_client()
    project_id = os.environ["CDSW_PROJECT_ID"]
    model = next(
        (m for m in client.list_models(project_id).models if m.name == MODEL_NAME), None
    )
    if model is None:
        raise RuntimeError(
            f"No deployed model named '{MODEL_NAME}'. Run scripts/deploy_model.py, "
            "or pass --local to use the local model."
        )

    resp = requests.post(
        f"https://modelservice.{os.environ['CDSW_DOMAIN']}/model",
        json={
            "accessKey": client.get_model(project_id, model.id).access_key,
            "request": {"dataframe_split": {"data": payload.tolist()}},
        },
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success"):
        raise RuntimeError(f"Endpoint returned an error: {body}")
    return np.asarray(body["response"]["prediction"], dtype=float)


def max_return_to_cvar(returns_dict, cvar_params):
    """Largest single-stock (mean return / CVaR) ratio for `returns_dict`.

    The same ratio the package's built-in risk-aversion scaling computes
    (CVaR._scale_risk_aversion), exposed here so it can be taken from
    historical data and held fixed while the forecasts change.
    """
    from portfolio_optimization import cvar_optimizer, cvar_utils
    from portfolio_optimization.settings import ApiSettings

    problem = cvar_optimizer.CVaR(
        returns_dict, cvar_params, ApiSettings(scale_risk_aversion=False)
    )
    single = cvar_utils.evaluate_single_asset_portfolios(problem)
    return float((single["return"] / single["CVaR"]).max())


def report_portfolio(label, result_row, portfolio, tickers, min_weight=0.005):
    print(f"\n--- {label} ---")
    weights = np.asarray(portfolio.weights, dtype=float)
    held = [(t, w) for t, w in zip(tickers, weights) if abs(w) >= min_weight]
    held.sort(key=lambda kv: -abs(kv[1]))
    print(f"  positions >= {min_weight:.1%}: {len(held)} of {len(tickers)}")
    for t, w in held:
        print(f"    {t:6s} {w:+7.2%}")
    print(f"  cash: {float(np.asarray(portfolio.cash).ravel()[0]):+.2%}")
    print(f"  sum of weights: {weights.sum():.4f}")
    for key in ("return", "CVaR", "obj", "solve time"):
        if key in result_row:
            print(f"  {key}: {result_row[key]}")


def main():
    args = parse_args()
    cp = require_gpu_stack()

    from portfolio_optimization import cvar_optimizer, cvar_utils, utils
    from portfolio_optimization.cvar_parameters import CvarParameters
    from portfolio_optimization.settings import (
        ApiSettings, KDESettings, ReturnsComputeSettings, ScenarioGenerationSettings,
    )
    from portfolio_optimization.forecasting.config import ForecastingConfig
    from portfolio_optimization.forecasting.garch_model import CovarianceForecaster
    from portfolio_optimization.forecasting.lightgbm_model import ReturnsForecaster
    from portfolio_optimization.forecasting.scenario_adjustment import rescale_scenarios

    cfg = ForecastingConfig()
    tmap = json.loads((PROJECT_ROOT / "models" / "ticker_categories.json").read_text())
    feature_order, ticker_codes = tmap["feature_order"], tmap["ticker_codes"]

    print("=== Forecast-driven Mean-CVaR optimization ===")
    print(f"  forecast horizon: {cfg.forecast_horizon} trading day(s)")
    print(f"  returns source:   {'local LightGBM' if args.local else 'deployed endpoint'}")

    # ---------------------------------------------------------------- prices
    print("\n--- Market data ---")
    prices = load_prices(sorted(ticker_codes), args.start, args.end)
    print(f"  {prices.shape[0]} trading days x {prices.shape[1]} tickers")
    print(f"  {prices.index[0].date()} -> {prices.index[-1].date()}")
    if len(prices) < GARCH_ROWS:
        print(f"  WARNING: only {len(prices)} rows; GARCH prefers >= {GARCH_ROWS}.")

    # -------------------------------------------------------------- forecasts
    print("\n--- Forecasts ---")
    if args.local:
        rf = ReturnsForecaster(cfg)
        rf.load(str(PROJECT_ROOT / "models" / "returns_forecaster.lgb"))
        predicted_returns = rf.predict(prices.tail(FEATURE_ROWS))
    else:
        predicted_returns = predict_returns_endpoint(prices, feature_order, ticker_codes)
    print(f"  expected returns: mean {predicted_returns.mean() * 100:+.4f}%, "
          f"spread(std) {predicted_returns.std() * 100:.4f}%, "
          f"{len(np.unique(predicted_returns.round(10)))} unique of {len(predicted_returns)}")

    cov_forecaster = CovarianceForecaster(cfg)
    cov_forecaster.train(prices.tail(GARCH_ROWS))
    covariance = cov_forecaster.predict(prices.tail(GARCH_ROWS))
    daily_vol = cov_forecaster.predict_volatility(prices.tail(GARCH_ROWS))
    print(f"  GARCH daily vol: min {daily_vol.min() * 100:.2f}%, "
          f"median {np.median(daily_vol) * 100:.2f}%, max {daily_vol.max() * 100:.2f}%")

    # ------------------------------------------------- returns_dict + scenarios
    print("\n--- Returns and scenarios ---")
    returns_dict = utils.calculate_returns(
        prices, None, ReturnsComputeSettings(return_type="LOG", freq=1)
    )
    tickers = returns_dict["tickers"]

    # Both models have already run above, so write their output straight in.
    # This is what ForecastClient.update_returns_dict() does internally; doing
    # it here keeps the endpoint path and the local path on one code path.
    forecast_dict = returns_dict.copy()
    forecast_dict["mean"] = predicted_returns
    forecast_dict["covariance"] = covariance
    assert not np.allclose(returns_dict["mean"], forecast_dict["mean"]), \
        "forecast mean identical to historical mean -- check the forecast step"

    bandwidth = args.kde_bandwidth or DEFAULT_BANDWIDTH[args.scenario_generator]
    scenario_settings = ScenarioGenerationSettings(
        num_scen=args.num_scen,
        fit_type="kde",
        kde_settings=KDESettings(
            bandwidth=bandwidth, kernel="gaussian", device="GPU",
            whiten=args.scenario_generator == "whitened-kde",
        ),
        seed=42,
    )
    print(f"  scenario generator: {args.scenario_generator}, bandwidth {bandwidth}")

    def make_scenarios(d):
        # Fits historical returns (d["returns"]) with a fixed seed, so every
        # call yields the same scenarios; d["mean"] only sets CvarData.mean.
        return cvar_utils.generate_cvar_data(d, scenario_settings)

    # NOTE ordering: the forecast mean must be set BEFORE this call, because
    # CVaR reads returns_dict["cvar_data"], not returns_dict["mean"]. The
    # scenarios come from historical returns; the GARCH forecast reaches them
    # only through rescale_scenarios below.
    forecast_dict = make_scenarios(forecast_dict)

    # Fidelity check: do the generated scenarios match actual history?
    hist_returns = np.asarray(returns_dict["returns"], dtype=float)
    hist_std = hist_returns.std(axis=0)
    upper = np.triu_indices(hist_returns.shape[1], 1)
    scen_R = forecast_dict["cvar_data"].R
    print(f"  scenario fidelity: volatility {np.median(scen_R.std(axis=1) / hist_std):.2f}x "
          f"history (median stock); average correlation "
          f"{np.corrcoef(scen_R)[upper].mean():.3f} vs history "
          f"{np.corrcoef(hist_returns, rowvar=False)[upper].mean():.3f}")

    if args.scenario_adjustment == "forecast":
        forecast_dict = rescale_scenarios(forecast_dict, predicted_returns, daily_vol)
        ratio = daily_vol / hist_std
        print("  scenarios rescaled to forecast mean and GARCH volatility")
        print(f"  GARCH vol / historical vol: min {ratio.min():.2f}x, "
              f"median {np.median(ratio):.2f}x, max {ratio.max():.2f}x")
    else:
        print("  scenarios NOT rescaled: CVaR uses historical KDE scenarios")
    cvar_data = forecast_dict["cvar_data"]
    print(f"  scenarios R: {cvar_data.R.shape} (assets x scenarios)")
    assert np.allclose(cvar_data.mean, predicted_returns), \
        "CvarData.mean is not the forecast -- ordering bug"
    print("  CvarData.mean is the forecast: True")

    # ----------------------------------------------------------------- solve
    print("\n--- Mean-CVaR solve (cuOpt) ---")
    cvar_params = CvarParameters(
        w_min=0.0,                 # long only
        w_max=args.w_max,          # position cap
        c_min=0.0,
        c_max=0.0,                 # fully invested
        risk_aversion=args.risk_aversion,
        confidence=args.confidence,
        cvar_limit=args.cvar_limit,
    )
    print(f"  w in [0, {args.w_max}], confidence {args.confidence}, "
          f"risk_aversion {args.risk_aversion}, cvar_limit {args.cvar_limit}")

    # Lambda scaling (see docs/forecasting_design_decisions.md). The package
    # rescales lambda by max_i(mean_i / CVaR_i) by default, computed from
    # whatever means and scenarios it is given. Computed from forecasts, that
    # ratio moves with the forecast level and cancels it out, so by default we
    # compute it from historical data and pass the already-scaled lambda in.
    # make_scenarios uses a fixed seed, so hist_dict holds the same
    # scenarios as forecast_dict; only the mean differs.
    hist_dict = make_scenarios(returns_dict)
    if args.lambda_scaling == "historical":
        scalar = max_return_to_cvar(hist_dict, cvar_params)
        cvar_params = cvar_params.model_copy(
            update={"risk_aversion": args.risk_aversion * scalar}
        )
        api_settings = ApiSettings(scale_risk_aversion=False)
    elif args.lambda_scaling == "forecast":
        api_settings = ApiSettings(scale_risk_aversion=True)
    else:
        api_settings = ApiSettings(scale_risk_aversion=False)

    solver_settings = {"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"}
    optimizer = cvar_optimizer.CVaR(forecast_dict, cvar_params, api_settings)
    if args.cvar_limit is None:
        print(f"  lambda scaling: {args.lambda_scaling}; lambda used in objective: "
              f"{optimizer.params.risk_aversion:.6g}")
    result_row, portfolio = optimizer.solve_optimization_problem(
        solver_settings=solver_settings, print_results=False
    )
    report_portfolio("Forecast-driven portfolio", result_row, portfolio, tickers)

    # -------------------------------------------------- optional comparison
    if args.historical_baseline:
        print("\n--- Baseline: historical mean/covariance ---")
        hist_optimizer = cvar_optimizer.CVaR(hist_dict, cvar_params, api_settings)
        if args.cvar_limit is None:
            print(f"  lambda used in objective: {hist_optimizer.params.risk_aversion:.6g}")
        hist_result, hist_portfolio = hist_optimizer.solve_optimization_problem(
            solver_settings=solver_settings, print_results=False
        )
        report_portfolio("Historical-baseline portfolio", hist_result,
                         hist_portfolio, tickers)
        turnover = np.abs(
            np.asarray(portfolio.weights) - np.asarray(hist_portfolio.weights)
        ).sum()
        print(f"\n  L1 weight difference between the two: {turnover:.4f}")
        print("  (near zero means the forecast barely changed the allocation)")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
