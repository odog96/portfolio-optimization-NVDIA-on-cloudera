from typing import Optional

import numpy as np
import pandas as pd

from portfolio_optimization.forecasting.config import ForecastingConfig
from portfolio_optimization.forecasting.garch_model import CovarianceForecaster
from portfolio_optimization.forecasting.lightgbm_model import ReturnsForecaster


class ForecastClient:
    """Unified client for returns and covariance forecasting.

    Supports two modes:
    1. Local: uses in-memory trained models directly.
    2. CAII: calls a deployed CAII endpoint for predictions.
    """

    def __init__(
        self,
        config: ForecastingConfig | None = None,
        returns_model: ReturnsForecaster | None = None,
        covariance_model: CovarianceForecaster | None = None,
    ):
        self.config = config or ForecastingConfig()
        self.returns_model = returns_model
        self.covariance_model = covariance_model

    def predict_returns(self, prices: pd.DataFrame) -> np.ndarray:
        """Get forward-looking expected returns vector.

        Parameters
        ----------
        prices : pd.DataFrame
            Recent price history for feature computation.

        Returns
        -------
        np.ndarray
            Predicted mean returns, shape (n_assets,).
        """
        if self.config.caii_endpoint:
            return self._call_caii_returns(prices)

        if self.returns_model is None:
            raise RuntimeError(
                "No returns model loaded and no CAII endpoint configured."
            )
        return self.returns_model.predict(prices)

    def predict_covariance(self, prices: pd.DataFrame) -> np.ndarray:
        """Get forward-looking covariance matrix.

        Parameters
        ----------
        prices : pd.DataFrame
            Recent price history for GARCH conditioning.

        Returns
        -------
        np.ndarray
            Predicted covariance matrix, shape (n_assets, n_assets).
        """
        if self.config.caii_endpoint:
            return self._call_caii_covariance(prices)

        if self.covariance_model is None:
            raise RuntimeError(
                "No covariance model loaded and no CAII endpoint configured."
            )
        return self.covariance_model.predict(prices)

    def predict(
        self, prices: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get both predicted returns and covariance.

        Returns
        -------
        mean : np.ndarray
            shape (n_assets,)
        covariance : np.ndarray
            shape (n_assets, n_assets)
        """
        mean = self.predict_returns(prices)
        cov = self.predict_covariance(prices)
        return mean, cov

    def update_returns_dict(
        self, returns_dict: dict, prices: pd.DataFrame
    ) -> dict:
        """Replace historical estimates in returns_dict with forecasts.

        This is the main integration point with the existing optimizer pipeline.
        Call this after calculate_returns() and before the optimizer.

        Parameters
        ----------
        returns_dict : dict
            Output of portfolio_optimization.utils.calculate_returns().
        prices : pd.DataFrame
            Recent price history for forecasting.

        Returns
        -------
        dict
            Updated returns_dict with forecasted mean and covariance.
        """
        mean, cov = self.predict(prices)
        returns_dict = returns_dict.copy()
        returns_dict["mean"] = mean
        returns_dict["covariance"] = cov
        return returns_dict

    def _call_caii_returns(self, prices: pd.DataFrame) -> np.ndarray:
        """Call CAII endpoint for returns prediction."""
        import requests

        payload = self._build_payload(prices)
        response = requests.post(
            f"{self.config.caii_endpoint}/predict_returns",
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )
        response.raise_for_status()
        return np.array(response.json()["mean_returns"])

    def _call_caii_covariance(self, prices: pd.DataFrame) -> np.ndarray:
        """Call CAII endpoint for covariance prediction."""
        import requests

        payload = self._build_payload(prices)
        response = requests.post(
            f"{self.config.caii_endpoint}/predict_covariance",
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )
        response.raise_for_status()
        return np.array(response.json()["covariance_matrix"])

    def _build_payload(self, prices: pd.DataFrame) -> dict:
        """Build JSON payload for CAII endpoint."""
        return {
            "tickers": prices.columns.tolist(),
            "dates": prices.index.strftime("%Y-%m-%d").tolist(),
            "prices": prices.values.tolist(),
            "forecast_horizon": self.config.forecast_horizon,
        }

    def _get_headers(self) -> dict:
        """Get authentication headers for CAII."""
        import os

        headers = {"Content-Type": "application/json"}
        token = os.environ.get("CAII_API_KEY", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
