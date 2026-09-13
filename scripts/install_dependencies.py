import subprocess
import sys


def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)


print("=== Installing Portfolio Optimization Dependencies ===")
print(f"Python: {sys.version}")

# Upgrade numpy first — CAI ships 1.26.4, project needs >=2.0.0.
# Must precede other installs to avoid binary incompatibility.
run("pip install --upgrade 'numpy>=2.0.0' 'pandas>=2.0'")

# Install all dependencies from requirements.txt
run("pip install -r /home/cdsw/portfolio-optimization/requirements.txt")

# Install the portfolio-optimization package in editable mode
run("pip install -e /home/cdsw/portfolio-optimization")

# Notebook kernel
run(
    "python3 -m ipykernel install --user "
    "--name=portfolio-opt "
    '--display-name "Portfolio Optimization"'
)

print("=== Dependencies installed ===")
