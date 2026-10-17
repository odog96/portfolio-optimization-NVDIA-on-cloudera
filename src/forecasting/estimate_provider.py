"""Forecast estimates for the rebalancing loop, without look-ahead.

`ForecastEstimateProvider` plugs the bundled LightGBM (expected returns) and
GARCH (volatility) models into `rebalance.rebalance_portfolio` through its
`estimate_provider` hook. Design: docs/forecasting_design_decisions.md,
section 5.

- `fit_schedule` trains the models before the backtest starts, then again
  every `retrain_months`, each time on prices strictly before that date.
- At a rebalance date, the latest models trained on or before it forecast
  from prices up to that date only.
- The forecast replaces the expected returns, the scenarios are rescaled to
  the forecast mean and volatility, and risk aversion is scaled from the
  historical estimates (sections 1 and 2).

The provider holds only fitted models and is picklable, so it can be trained
once and shared with worker processes.
"""

import pickle
import time

import numpy as np
import pandas as pd

from portfolio_optimization.forecasting.config import ForecastingConfig
from portfolio_optimization.forecasting.garch_model import CovarianceForecaster
from portfolio_optimization.forecasting.lightgbm_model import ReturnsForecaster
from portfolio_optimization.forecasting.scenario_adjustment import rescale_scenarios

FEATURE_ROWS = 64  # rows needed for one LightGBM feature row per ticker
GARCH_ROWS = 500  # window for the GARCH fits and volatility forecasts


def max_return_to_cvar(returns_dict, cvar_params):
    """Largest single-asset (mean return / CVaR) ratio for `returns_dict`.

    The ratio the CVaR optimizer's built-in risk-aversion scaling computes,
    exposed so it can be taken from historical estimates and held fixed while
    the forecasts change (design doc section 1).
    """
    from portfolio_optimization import cvar_optimizer, cvar_utils
    from portfolio_optimization.settings import ApiSettings

    problem = cvar_optimizer.CVaR(
        returns_dict, cvar_params, ApiSettings(scale_risk_aversion=False)
    )
    single = cvar_utils.evaluate_single_asset_portfolios(problem)
    return float((single["return"] / single["CVaR"]).max())


class ForecastEstimateProvider:
    """Walk-forward forecast estimates for `rebalance_portfolio`.

    Parameters
    ----------
    config : ForecastingConfig, optional
        Model settings. `return_type` must match the returns the optimizer
        uses (LOG or LINEAR).
    retrain_months : int
        Months between retrainings during the backtest.
    """

    def __init__(self, config: ForecastingConfig | None = None, retrain_months: int = 3):
        self.config = config or ForecastingConfig()
        self.retrain_months = retrain_months
        self.models: list[tuple[pd.Timestamp, ReturnsForecaster, CovarianceForecaster]] = []
        self.tickers: list[str] = []
        self.training_time = 0.0
        self.last_forecast_time = 0.0

    def fit_schedule(self, prices: pd.DataFrame, start, end) -> "ForecastEstimateProvider":
        """Train at `start` and every `retrain_months` until `end`.

        Each training uses only prices dated strictly before its training
        date: the last `config.training_window` rows for LightGBM and the last
        GARCH_ROWS rows for GARCH.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.tickers = prices.columns.tolist()
        t0 = time.time()
        self.models = []
        train_date = start
        while train_date <= end:
            history = prices.loc[prices.index < train_date]
            if len(history) < max(self.config.training_window, GARCH_ROWS) // 2:
                raise ValueError(
                    f"Only {len(history)} price rows before {train_date.date()}; "
                    "choose a later backtest start."
                )
            returns_model = ReturnsForecaster(self.config)
            returns_model.train(history.tail(self.config.training_window))
            vol_model = CovarianceForecaster(self.config)
            vol_model.train(history.tail(GARCH_ROWS))
            self.models.append((train_date, returns_model, vol_model))
            train_date = train_date + pd.DateOffset(months=self.retrain_months)
        self.training_time = time.time() - t0
        return self

    def models_for(self, date):
        """The latest (training date, returns model, volatility model) on or before `date`."""
        date = pd.Timestamp(date)
        usable = [m for m in self.models if m[0] <= date]
        if not usable:
            raise ValueError(
                f"No model trained on or before {date.date()}; the first "
                f"training date is {self.models[0][0].date() if self.models else 'none'}."
            )
        return usable[-1]

    def __call__(self, returns_dict, cvar_params, prices, date):
        """Estimate hook for `rebalance_portfolio`.

        Returns the updated returns_dict, cvar_params with the risk aversion
        already scaled, and ApiSettings that turn off the optimizer's own
        scaling.
        """
        from portfolio_optimization.settings import ApiSettings

        t0 = time.time()
        _, returns_model, vol_model = self.models_for(date)
        history = prices.loc[prices.index <= pd.Timestamp(date), self.tickers]

        # Predict over the training tickers (LightGBM encodes the ticker as a
        # category in training order), then pick the optimizer's tickers.
        mean = pd.Series(returns_model.predict(history.tail(FEATURE_ROWS)), index=self.tickers)
        vol = pd.Series(vol_model.predict_volatility(history.tail(GARCH_ROWS)), index=self.tickers)
        tickers = list(returns_dict["tickers"])

        scalar = max_return_to_cvar(returns_dict, cvar_params)
        updated = rescale_scenarios(
            returns_dict, mean[tickers].to_numpy(), vol[tickers].to_numpy()
        )
        params = cvar_params.model_copy(
            update={"risk_aversion": cvar_params.risk_aversion * scalar}
        )
        self.last_forecast_time = time.time() - t0
        return updated, params, ApiSettings(scale_risk_aversion=False)

    def save(self, path: str) -> str:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: str) -> "ForecastEstimateProvider":
        with open(path, "rb") as f:
            return pickle.load(f)
