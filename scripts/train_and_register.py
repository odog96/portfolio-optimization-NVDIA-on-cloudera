"""Train forecasting models, log to MLflow, register in CML model registry.

Follows the cmlapi pattern from CAI-baseline-workshop/module1/04_deploy.py:
  1. Train models
  2. Log native LightGBM model to MLflow experiment run (with signature)
  3. Register in CML via cmlapi.CreateRegisteredModelRequest
"""
import json
import os
import sys
from pathlib import Path

# Cloudera AI runs scripts cell by cell in a kernel where __file__ is undefined;
# there the working directory is the project root.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import numpy as np
from mlflow.models import infer_signature

from portfolio_optimization.forecasting.config import ForecastingConfig
from portfolio_optimization.forecasting.garch_model import CovarianceForecaster
from portfolio_optimization.forecasting.lightgbm_model import ReturnsForecaster
from portfolio_optimization.forecasting.feature_engineering import (
    build_training_data,
    flatten_features_for_training,
)
from portfolio_optimization.utils import get_input_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
dataset = os.environ.get("PORTFOLIO_OPT_DATASET", "dow30")
data_path = str(PROJECT_ROOT / "data" / "stock_data" / f"{dataset}.csv")
model_dir = str(PROJECT_ROOT / "models")
os.makedirs(model_dir, exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "PortfolioReturnsForecaster"
EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "portfolio_returns_forecasting"
)

print(f"=== Training Forecasting Models ({dataset}) ===")

prices = get_input_data(data_path)
print(f"Loaded {prices.shape[1]} assets, {prices.shape[0]} trading days")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

# forecast_horizon=1 keeps the forecast mean on the same daily scale as the
# GARCH covariance and the CVaR scenarios (which are built from freq=1 daily
# returns). Changing it requires rescaling both — see docs/forecasting_design_decisions.md, section 4.
FORECAST_HORIZON = int(os.environ.get("FORECAST_HORIZON", "1"))
config = ForecastingConfig(
    forecast_horizon=FORECAST_HORIZON, training_window=756, return_type="LOG"
)

# ---------------------------------------------------------------------------
# Step 1: Train LightGBM
# ---------------------------------------------------------------------------
print("\n--- Training LightGBM returns forecaster ---")
returns_forecaster = ReturnsForecaster(config)
metrics = returns_forecaster.train(prices)
print(f"  Horizon:    {config.forecast_horizon} trading day(s)")
print(f"  Split:      train <= {metrics['train_end']} | eval >= {metrics['eval_start']}")
print(f"  Train RMSE: {metrics['train_rmse']:.6f}")
print(f"  Eval RMSE:  {metrics['eval_rmse']:.6f}  (out-of-sample, chronological)")
print(f"  Features:   {metrics['n_features']}")
print(f"  Samples:    {metrics['n_samples']} "
      f"({metrics['n_train_rows']} train / {metrics['n_eval_rows']} eval)")
print("  --- skill diagnostics (read these before trusting the forecast) ---")
print(f"  Baseline RMSE:        {metrics['baseline_rmse']:.6f}  "
      f"(predict the training mean for every asset)")
print(f"  Skill vs baseline:    {metrics['skill_vs_baseline']:+.4%}  "
      f"(<= 0 means no better than the baseline)")
print(f"  Directional accuracy: {metrics['directional_accuracy']:.4f}  "
      f"(0.50 = coin flip)")
print(f"  Prediction spread:    {metrics['pred_std']:.6f} vs actual "
      f"{metrics['actual_std']:.6f} "
      f"({metrics['pred_std'] / metrics['actual_std']:.1%} of actual)")
if metrics["skill_vs_baseline"] < 0.01:
    print("  NOTE: skill is under 1% — this forecast carries little information.")
    print("        See docs/forecasting_design_decisions.md, section 4.")

lgb_path = returns_forecaster.save(f"{model_dir}/returns_forecaster.lgb")
print(f"  Saved native: {lgb_path}")

# ---------------------------------------------------------------------------
# Step 2: Train GARCH
# ---------------------------------------------------------------------------
print("\n--- Training GARCH covariance forecaster ---")
cov_forecaster = CovarianceForecaster(config)
garch_metrics = cov_forecaster.train(prices)
converged = sum(1 for m in garch_metrics.values() if m.get("converged"))
print(f"  Converged: {converged}/{len(garch_metrics)}")

garch_path = cov_forecaster.export_params(f"{model_dir}/garch_params.json")
print(f"  Exported params: {garch_path}")

# ---------------------------------------------------------------------------
# Step 3: Log LightGBM model to MLflow with signature
# ---------------------------------------------------------------------------
print("\n--- Logging LightGBM model to MLflow ---")

mlflow.set_experiment(EXPERIMENT_NAME)

# Build signature from sample data
X, y = build_training_data(
    prices,
    forecast_horizon=config.forecast_horizon,
    return_type=config.return_type,
    config=config.features,
)
X_flat, y_flat = flatten_features_for_training(X, y)
# The model has 12 features: 11 numeric + `ticker`, a categorical that the
# trees split on by integer category code. The served signature must cover all
# 12 columns (ticker as its code) and be float64 — JSON numbers always arrive
# as float64, so a float32 tensor spec can never be satisfied over HTTP.
ticker_categories = X_flat["ticker"].cat.categories.tolist()
serving_frame = X_flat.assign(ticker=X_flat["ticker"].cat.codes)
sample_input = serving_frame.iloc[:5].values.astype(np.float64)
sample_preds = returns_forecaster.model.predict(X_flat.iloc[:5])
signature = infer_signature(sample_input, sample_preds)
ticker_map_path = os.path.join(model_dir, "ticker_categories.json")
with open(ticker_map_path, "w") as f:
    json.dump(
        {
            "feature_order": serving_frame.columns.tolist(),
            "ticker_codes": {t: i for i, t in enumerate(ticker_categories)},
        },
        f,
        indent=2,
    )

with mlflow.start_run(run_name="train_returns_forecaster") as run:
    mlflow.lightgbm.log_model(
        returns_forecaster.model,
        artifact_path="model",
        signature=signature,
    )
    mlflow.log_metrics({
        "train_rmse": metrics["train_rmse"],
        "eval_rmse": metrics["eval_rmse"],
        "baseline_rmse": metrics["baseline_rmse"],
        "skill_vs_baseline": metrics["skill_vs_baseline"],
        "directional_accuracy": metrics["directional_accuracy"],
        "pred_std": metrics["pred_std"],
        "actual_std": metrics["actual_std"],
    })
    mlflow.log_param("forecast_horizon", config.forecast_horizon)
    mlflow.log_artifact(garch_path)
    mlflow.log_artifact(ticker_map_path)
    mlflow.set_tag("model_type", "lightgbm_native")
    mlflow.set_tag("dataset", dataset)

run_id = run.info.run_id
experiment_id = run.info.experiment_id
print(f"  MLflow run: {run_id}")
print(f"  Experiment: {experiment_id}")

# ---------------------------------------------------------------------------
# Step 4: Register in CML model registry via cmlapi
# ---------------------------------------------------------------------------
print("\n--- Registering model in CML registry ---")

project_id = os.environ.get("CDSW_PROJECT_ID")

if not project_id:
    print("  CDSW_PROJECT_ID not set — skipping CML registration.")
    print("  (Model logged to MLflow — register manually via UI)")
else:
    try:
        import cmlapi
        from cmlapi.rest import ApiException

        cml_client = cmlapi.default_client()

        # Check if model already exists in registry
        existing_models = cml_client.list_registered_models()
        existing_model = None
        if hasattr(existing_models, "models") and existing_models.models:
            for m in existing_models.models:
                if m.name == MODEL_NAME:
                    existing_model = m
                    break

        if existing_model:
            print(f"  Model '{MODEL_NAME}' exists: {existing_model.model_id}")
            print(f"  Deleting old registration to re-register with new run...")
            cml_client.delete_registered_model(existing_model.model_id)

        reg_req = cmlapi.CreateRegisteredModelRequest(
            project_id=project_id,
            experiment_id=experiment_id,
            run_id=run_id,
            model_name=MODEL_NAME,
            model_path="model",
        )
        reg_resp = cml_client.create_registered_model(body=reg_req)
        registered_model_id = reg_resp.model_id
        model_version_id = reg_resp.model_versions[0].model_version_id
        print(f"  Registered: {registered_model_id}")
        print(f"  Version: {model_version_id}")

        # Save for deploy script
        deploy_info = {
            "model_name": MODEL_NAME,
            "registered_model_id": registered_model_id,
            "model_version_id": str(model_version_id),
            "run_id": run_id,
            "experiment_id": experiment_id,
            "eval_rmse": metrics["eval_rmse"],
        }
        with open(OUTPUT_DIR / "registration_info.json", "w") as f:
            json.dump(deploy_info, f, indent=2)
        print(f"  Saved: {OUTPUT_DIR / 'registration_info.json'}")

    except ApiException as e:
        print(f"  CML API error: {e.reason}")
        print(f"  {e.body}")
    except ImportError:
        print("  cmlapi not available — skip CML registration.")
    except Exception as e:
        print(f"  Registration failed: {e}")

print("\n=== Training and registration complete ===")
