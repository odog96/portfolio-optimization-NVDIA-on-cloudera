import os
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
    2. CAII: calls a deployed CAII endpoint via OpenInferenceClient.
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
        self._inference_client = None

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
        if self.config.caii_covariance_endpoint:
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

    def _get_inference_client(self):
        """Get or create an OpenInferenceClient for CAII."""
        if self._inference_client is not None:
            return self._inference_client

        try:
            from open_inference.openapi.client import OpenInferenceClient
        except ImportError:
            raise ImportError(
                "Install the CAII client: pip install open_inference_client"
            )

        import httpx

        api_key = os.environ.get("CAII_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "CAII_API_KEY environment variable is required for CAII mode."
            )

        self._inference_client = OpenInferenceClient(
            base_url=self.config.caii_endpoint,
            httpx_client=httpx.Client(
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            ),
        )
        return self._inference_client

    def _call_caii_returns(self, prices: pd.DataFrame) -> np.ndarray:
        """Call CAII endpoint for returns prediction via OpenInferenceClient."""
        from portfolio_optimization.forecasting.feature_engineering import (
            compute_features,
        )

        features = compute_features(prices, self.config.features)
        tickers = prices.columns.tolist()

        input_rows = []
        for ticker in tickers:
            if ticker in features.columns.get_level_values(0):
                ticker_feats = features[ticker].iloc[-1].values.astype(
                    np.float32
                )
                input_rows.append(ticker_feats)
            else:
                input_rows.append(
                    np.zeros(features[tickers[0]].shape[1], dtype=np.float32)
                )

        input_array = np.array(input_rows, dtype=np.float32)

        client = self._get_inference_client()
        response = client.infer(
            model_name="PortfolioReturnsForecaster",
            inputs={"input": input_array},
        )

        return np.array(response.outputs[0].data, dtype=np.float64)

    def _call_caii_covariance(self, prices: pd.DataFrame) -> np.ndarray:
        """Call CAII endpoint for covariance prediction.

        Falls back to local GARCH if no separate covariance endpoint is set,
        since GARCH params are not natively ONNX-servable.
        """
        if self.covariance_model is not None:
            return self.covariance_model.predict(prices)

        import requests

        payload = self._build_payload(prices)
        response = requests.post(
            f"{self.config.caii_covariance_endpoint}/predict_covariance",
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
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("CAII_API_KEY", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
