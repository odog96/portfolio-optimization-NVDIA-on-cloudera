from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from portfolio_optimization.forecasting.config import ForecastingConfig, GARCHConfig


class CovarianceForecaster:
    """GARCH-based covariance matrix forecaster.

    Fits univariate GARCH models per asset, then combines with a DCC-like
    correlation structure to produce a forward-looking covariance matrix.
    """

    def __init__(self, config: ForecastingConfig | None = None):
        self.config = config or ForecastingConfig()
        self.garch_models: dict[str, object] = {}
        self.tickers: list[str] = []
        self._correlation_matrix: np.ndarray | None = None

    def train(self, prices: pd.DataFrame) -> dict:
        """Fit GARCH models on historical returns.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (date index, ticker columns).

        Returns
        -------
        dict
            Training metrics per asset: converged, AIC, BIC.
        """
        from arch import arch_model

        self.tickers = prices.columns.tolist()
        returns = np.log(prices / prices.shift(1)).dropna() * 100  # scale for arch

        metrics = {}
        for ticker in self.tickers:
            ret = returns[ticker].dropna()
            garch_cfg = self.config.garch

            am = arch_model(
                ret,
                vol=garch_cfg.vol_model,
                p=garch_cfg.p,
                q=garch_cfg.q,
                mean=garch_cfg.mean_model,
                dist=garch_cfg.dist,
            )

            try:
                res = am.fit(disp="off", show_warning=False)
                self.garch_models[ticker] = res
                metrics[ticker] = {
                    "converged": res.convergence_flag == 0,
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    "persistence": float(
                        sum(res.params.get(f"alpha[{i+1}]", 0) for i in range(garch_cfg.q))
                        + sum(res.params.get(f"beta[{i+1}]", 0) for i in range(garch_cfg.p))
                    ),
                }
            except Exception as e:
                metrics[ticker] = {"converged": False, "error": str(e)}

        # Compute unconditional correlation from standardized residuals
        self._correlation_matrix = self._estimate_correlation(returns)

        return metrics

    def predict(
        self, prices: pd.DataFrame, horizon: int | None = None
    ) -> np.ndarray:
        """Forecast the covariance matrix.

        Parameters
        ----------
        prices : pd.DataFrame
            Recent price history for conditioning the GARCH forecasts.
        horizon : int, optional
            Forecast horizon in trading days. Defaults to config.forecast_horizon.

        Returns
        -------
        np.ndarray
            Forecasted covariance matrix, shape (n_assets, n_assets).
        """
        if not self.garch_models:
            raise RuntimeError("Models not trained. Call train() first.")

        horizon = horizon or self.config.forecast_horizon
        tickers = prices.columns.tolist()
        n = len(tickers)

        # Forecast variance per asset
        forecasted_vol = np.zeros(n)
        returns = np.log(prices / prices.shift(1)).dropna() * 100

        for i, ticker in enumerate(tickers):
            if ticker in self.garch_models:
                res = self.garch_models[ticker]
                forecast = res.forecast(horizon=horizon, reindex=False)
                # Average variance over the horizon, convert back from pct scale
                avg_var = forecast.variance.values[-1, :horizon].mean()
                forecasted_vol[i] = np.sqrt(avg_var) / 100
            else:
                forecasted_vol[i] = returns[ticker].std() / 100

        # Build covariance: D * R * D where D = diag(vol), R = correlation
        corr = self._get_correlation(tickers)
        D = np.diag(forecasted_vol)
        covariance = D @ corr @ D

        # Ensure positive semi-definite
        covariance = _nearest_psd(covariance)
        return covariance

    def export_params(self, path: str) -> str:
        """Export GARCH parameters and correlation matrix for ONNX deployment.

        Since GARCH is recursive and not natively ONNX-compatible, we export
        the fitted parameters as JSON. The CAII endpoint uses these to
        reconstruct and forecast.
        """
        import json

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "tickers": self.tickers,
            "garch_config": self.config.garch.model_dump(),
            "models": {},
        }

        for ticker, res in self.garch_models.items():
            params["models"][ticker] = {
                "params": {k: float(v) for k, v in res.params.items()},
                "conditional_volatility_last": float(
                    res.conditional_volatility.iloc[-1]
                ),
                "resid_last": float(res.resid.iloc[-1]),
            }

        if self._correlation_matrix is not None:
            params["correlation_matrix"] = self._correlation_matrix.tolist()

        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)

        return str(output_path)

    def register_mlflow(
        self,
        model_name: str = "PortfolioCovarianceForecaster",
    ) -> str:
        """Register GARCH parameters in MLflow model registry.

        GARCH is recursive and cannot be natively exported to ONNX, so we
        log the fitted parameters as a pyfunc model artifact. This bridges
        to CAII for covariance predictions via a custom wrapper.

        Parameters
        ----------
        model_name : str
            Registered model name in MLflow.

        Returns
        -------
        str
            Model version URI.
        """
        if not self.garch_models:
            raise RuntimeError("Models not trained.")

        import json
        import tempfile

        import mlflow
        import mlflow.pyfunc

        params = {
            "tickers": self.tickers,
            "garch_config": self.config.garch.model_dump(),
            "models": {},
        }
        for ticker, res in self.garch_models.items():
            params["models"][ticker] = {
                "params": {k: float(v) for k, v in res.params.items()},
                "conditional_volatility_last": float(
                    res.conditional_volatility.iloc[-1]
                ),
                "resid_last": float(res.resid.iloc[-1]),
            }
        if self._correlation_matrix is not None:
            params["correlation_matrix"] = self._correlation_matrix.tolist()

        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = Path(tmpdir) / "garch_params.json"
            with open(params_path, "w") as f:
                json.dump(params, f, indent=2)

            with mlflow.start_run(run_name="register_covariance_forecaster"):
                mlflow.log_artifact(str(params_path))
                result = mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=None,
                    artifacts={"garch_params": str(params_path)},
                    registered_model_name=model_name,
                )
                mlflow.set_tag("model_type", "garch_params")
                mlflow.set_tag("purpose", "covariance_forecasting")
                mlflow.set_tag("n_assets", str(len(self.tickers)))

        print(f"Registered '{model_name}' in MLflow: {result.model_uri}")
        return result.model_uri

    @classmethod
    def load_params(cls, path: str) -> "CovarianceForecaster":
        """Load exported parameters (for CAII deployment)."""
        import json

        with open(path) as f:
            params = json.load(f)

        forecaster = cls()
        forecaster.tickers = params["tickers"]
        if "correlation_matrix" in params:
            forecaster._correlation_matrix = np.array(
                params["correlation_matrix"]
            )
        return forecaster

    def _estimate_correlation(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate correlation from standardized residuals where available."""
        standardized = pd.DataFrame(index=returns.index)

        for ticker in self.tickers:
            if ticker in self.garch_models:
                res = self.garch_models[ticker]
                std_resid = res.resid / res.conditional_volatility
                standardized[ticker] = std_resid
            else:
                standardized[ticker] = (
                    returns[ticker] - returns[ticker].mean()
                ) / returns[ticker].std()

        corr = standardized.corr().values
        # Ensure valid correlation matrix
        np.fill_diagonal(corr, 1.0)
        corr = np.nan_to_num(corr, nan=0.0)
        return corr

    def _get_correlation(self, tickers: list[str]) -> np.ndarray:
        """Get correlation submatrix for the given tickers."""
        if self._correlation_matrix is None:
            return np.eye(len(tickers))

        indices = [
            self.tickers.index(t) if t in self.tickers else -1 for t in tickers
        ]
        n = len(tickers)
        corr = np.eye(n)
        for i in range(n):
            for j in range(n):
                if indices[i] >= 0 and indices[j] >= 0:
                    corr[i, j] = self._correlation_matrix[indices[i], indices[j]]
        return corr


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a matrix to the nearest positive semi-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
