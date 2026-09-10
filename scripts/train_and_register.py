"""Train forecasting models and register in MLflow for CAII deployment.

AMP task: runs after data download, before CAII deployment.
Trains LightGBM (returns) and GARCH (covariance) on the downloaded dataset,
exports to ONNX, and registers in MLflow model registry.
"""
import os
import sys

sys.path.insert(0, "/home/cdsw/portfolio-optimization")

from portfolio_optimization.forecasting.config import ForecastingConfig
from portfolio_optimization.forecasting.garch_model import CovarianceForecaster
from portfolio_optimization.forecasting.lightgbm_model import ReturnsForecaster
from portfolio_optimization.utils import get_input_data

dataset = os.environ.get("PORTFOLIO_OPT_DATASET", "dow30")
data_path = f"/home/cdsw/portfolio-optimization/data/stock_data/{dataset}.csv"

print(f"=== Training Forecasting Models ({dataset}) ===")

prices = get_input_data(data_path)
print(f"Loaded {prices.shape[1]} assets, {prices.shape[0]} trading days")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

config = ForecastingConfig(forecast_horizon=5, training_window=756, return_type="LOG")

# --- LightGBM (expected returns) ---
print("\n--- Training LightGBM returns forecaster ---")
returns_forecaster = ReturnsForecaster(config)
metrics = returns_forecaster.train(prices)
print(f"  Train RMSE: {metrics['train_rmse']:.6f}")
print(f"  Eval RMSE:  {metrics['eval_rmse']:.6f}")
print(f"  Features:   {metrics['n_features']}")
print(f"  Samples:    {metrics['n_samples']}")

model_dir = "/home/cdsw/portfolio-optimization/models"
os.makedirs(model_dir, exist_ok=True)

lgb_path = returns_forecaster.save(f"{model_dir}/returns_forecaster.lgb")
print(f"  Saved native: {lgb_path}")

try:
    onnx_path = returns_forecaster.export_onnx(f"{model_dir}/returns_forecaster.onnx")
    print(f"  Exported ONNX: {onnx_path}")
except Exception as e:
    print(f"  ONNX export failed: {e}")

# --- GARCH (covariance) ---
print("\n--- Training GARCH covariance forecaster ---")
cov_forecaster = CovarianceForecaster(config)
garch_metrics = cov_forecaster.train(prices)
converged = sum(1 for m in garch_metrics.values() if m.get("converged"))
print(f"  Converged: {converged}/{len(garch_metrics)}")

garch_path = cov_forecaster.export_params(f"{model_dir}/garch_params.json")
print(f"  Exported params: {garch_path}")

# --- MLflow Registration ---
print("\n--- Registering models in MLflow ---")
try:
    returns_uri = returns_forecaster.register_mlflow(
        model_name="PortfolioReturnsForecaster"
    )
    print(f"  LightGBM registered: {returns_uri}")
except Exception as e:
    print(f"  LightGBM MLflow registration failed: {e}")
    print("  (Model files saved locally — register manually if needed)")

try:
    cov_uri = cov_forecaster.register_mlflow(
        model_name="PortfolioCovarianceForecaster"
    )
    print(f"  GARCH registered: {cov_uri}")
except Exception as e:
    print(f"  GARCH MLflow registration failed: {e}")
    print("  (Params saved locally — register manually if needed)")

print("\n=== Training and registration complete ===")
