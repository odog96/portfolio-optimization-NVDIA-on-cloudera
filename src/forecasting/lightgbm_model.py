from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from portfolio_optimization.forecasting.config import (
    ForecastingConfig,
    LightGBMConfig,
)
from portfolio_optimization.forecasting.feature_engineering import (
    build_training_data,
    compute_features,
    flatten_features_for_training,
)


class ReturnsForecaster:
    """LightGBM-based expected returns forecaster.

    Trains a single model across all assets using per-asset technical features.
    Produces a forward-looking mean return vector for portfolio optimization.
    """

    def __init__(self, config: ForecastingConfig | None = None):
        self.config = config or ForecastingConfig()
        self.model = None
        self.feature_names: list[str] = []

    def train(
        self,
        prices: pd.DataFrame,
        eval_fraction: float = 0.15,
    ) -> dict:
        """Train LightGBM on historical price data.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (date index, ticker columns).
        eval_fraction : float
            Fraction of data to hold out for early stopping (time-sorted).

        Returns
        -------
        dict
            Training metrics: train_rmse, eval_rmse, n_features, n_samples.
        """
        import lightgbm as lgb

        X, y = build_training_data(
            prices,
            forecast_horizon=self.config.forecast_horizon,
            return_type=self.config.return_type,
            config=self.config.features,
        )
        X_flat, y_flat = flatten_features_for_training(X, y)

        self.feature_names = X_flat.columns.tolist()

        # Time-based split for evaluation
        split_idx = int(len(X_flat) * (1 - eval_fraction))
        X_train, X_eval = X_flat.iloc[:split_idx], X_flat.iloc[split_idx:]
        y_train, y_eval = y_flat.iloc[:split_idx], y_flat.iloc[split_idx:]

        lgb_cfg = self.config.lightgbm
        train_set = lgb.Dataset(X_train, label=y_train)
        eval_set = lgb.Dataset(X_eval, label=y_eval, reference=train_set)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": lgb_cfg.n_estimators,
            "max_depth": lgb_cfg.max_depth,
            "learning_rate": lgb_cfg.learning_rate,
            "subsample": lgb_cfg.subsample,
            "colsample_bytree": lgb_cfg.colsample_bytree,
            "min_child_samples": lgb_cfg.min_child_samples,
            "reg_alpha": lgb_cfg.reg_alpha,
            "reg_lambda": lgb_cfg.reg_lambda,
            "seed": lgb_cfg.seed,
            "verbose": -1,
        }

        callbacks = [
            lgb.early_stopping(lgb_cfg.early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            params,
            train_set,
            valid_sets=[eval_set],
            callbacks=callbacks,
        )

        train_preds = self.model.predict(X_train)
        eval_preds = self.model.predict(X_eval)

        return {
            "train_rmse": float(np.sqrt(np.mean((y_train - train_preds) ** 2))),
            "eval_rmse": float(np.sqrt(np.mean((y_eval - eval_preds) ** 2))),
            "n_features": len(self.feature_names),
            "n_samples": len(X_flat),
            "best_iteration": self.model.best_iteration,
        }

    def predict(self, prices: pd.DataFrame) -> np.ndarray:
        """Predict expected returns for each asset.

        Parameters
        ----------
        prices : pd.DataFrame
            Recent price history (must be long enough for feature computation).

        Returns
        -------
        np.ndarray
            Predicted mean return per asset, shape (n_assets,).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        features = compute_features(prices, self.config.features)
        tickers = prices.columns.tolist()
        predictions = {}

        for ticker in tickers:
            if ticker not in features.columns.get_level_values(0):
                predictions[ticker] = 0.0
                continue
            ticker_feats = features[ticker].iloc[[-1]].copy()
            ticker_feats["ticker"] = pd.Categorical(
                [ticker], categories=tickers
            )
            pred = self.model.predict(ticker_feats)
            predictions[ticker] = float(pred[0])

        return np.array([predictions[t] for t in tickers])

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from the trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

    def export_onnx(self, path: str) -> str:
        """Export trained model to ONNX format.

        Parameters
        ----------
        path : str
            Output file path (e.g., "models/returns_forecaster.onnx").

        Returns
        -------
        str
            Path to saved ONNX file.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        import onnxmltools
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,  # noqa: F401
        )
        from skl2onnx.common.data_types import FloatTensorType

        n_features = len(self.feature_names)
        initial_type = [("input", FloatTensorType([None, n_features]))]

        onnx_model = onnxmltools.convert_lightgbm(
            self.model, initial_types=initial_type, target_opset=15
        )

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnxmltools.utils.save_model(onnx_model, str(output_path))
        return str(output_path)

    def save(self, path: str) -> str:
        """Save native LightGBM model."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(output_path))
        return str(output_path)

    def load(self, path: str) -> None:
        """Load a previously saved LightGBM model."""
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=path)
