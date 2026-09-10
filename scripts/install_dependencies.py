import subprocess
import sys


def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)


print("=== Installing Portfolio Optimization Dependencies ===")
print(f"Python: {sys.version}")

# Upgrade numpy first — CAI ships 1.26.4, project needs >=2.0.0.
# Must precede other installs to avoid binary incompatibility.
run("pip install --upgrade 'numpy>=2.0.0'")

# Upgrade pandas for numpy 2.x binary compat
run("pip install --upgrade 'pandas>=2.0'")

# Core dependencies not pre-installed in the CAI runtime
run(
    "pip install "
    "'cvxpy>=1.9.2' "
    "'scikit-learn>=1.5' "
    "'seaborn>=0.13' "
    "'yfinance>=0.2.0' "
    "'pydantic>=2.12.5' "
    "'matplotlib>=3.8'"
)

# Forecasting dependencies
run(
    "pip install "
    "'lightgbm>=4.0' "
    "'arch>=7.0' "
    "'onnxmltools>=1.12' "
    "'skl2onnx>=1.16' "
    "'mlflow>=2.12'"
)

# Install the portfolio-optimization package in editable mode
run("pip install -e /home/cdsw/portfolio-optimization")

# Notebook kernel
run("pip install 'ipykernel>=7.1.0'")
run(
    "python3 -m ipykernel install --user "
    "--name=portfolio-opt "
    '--display-name "Portfolio Optimization"'
)

print("=== Core dependencies installed ===")
