import os
import subprocess

import numpy as np
import pandas as pd

# Show GPU info
print("=== GPU Validation ===")
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "nvidia-smi not available")
print(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")

# Verify GPU package imports
import cuml
import cuopt

print(f"cuML: {cuml.__version__}, cuOpt: {cuopt.__version__}")

# End-to-end smoke test
from portfolio_optimization.cvar_optimizer import CVaR
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.settings import (
    ApiSettings,
    KDESettings,
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)
from portfolio_optimization.utils import calculate_returns

np.random.seed(42)
dates = pd.bdate_range("2023-01-01", periods=250)
prices = pd.DataFrame(
    100 + np.cumsum(np.random.randn(250, 5) * 0.5, axis=0),
    index=dates,
    columns=["A", "B", "C", "D", "E"],
)

rd = calculate_returns(prices, returns_compute_settings=ReturnsComputeSettings())
rd = generate_cvar_data(
    rd,
    ScenarioGenerationSettings(
        num_scen=1000,
        fit_type="kde",
        kde_settings=KDESettings(device="GPU"),
        seed=42,
    ),
)

opt = CVaR(
    returns_dict=rd,
    cvar_params=CvarParameters(w_min=0.0, w_max=0.5, confidence=0.95),
    api_settings=ApiSettings(api="cuopt_python"),
)
result, portfolio = opt.solve_optimization_problem(print_results=True)

print("\n=== GPU pipeline validated successfully ===")
