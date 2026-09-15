import subprocess
import sys
from pathlib import Path

# Project root, derived from this script's location rather than hardcoded —
# the project directory name/path can differ across workspaces.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)


print("=== Installing Portfolio Optimization Dependencies ===")
print(f"Python: {sys.version}")

# Upgrade numpy first — CAI ships 1.26.4, project needs >=2.0.0.
# Must precede other installs to avoid binary incompatibility.
run("pip install --upgrade 'numpy>=2.0.0' 'pandas>=2.0'")

# Install all dependencies from requirements.txt
run(f"pip install -r {PROJECT_ROOT / 'requirements.txt'}")

# Install the portfolio-optimization package in editable mode
run(f"pip install -e {PROJECT_ROOT}")

# Notebook kernel
run(
    "python3 -m ipykernel install --user "
    "--name=portfolio-opt "
    '--display-name "Portfolio Optimization"'
)

print("=== Dependencies installed ===")
