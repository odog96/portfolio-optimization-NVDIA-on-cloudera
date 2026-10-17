import subprocess
import sys
from pathlib import Path

# Project root, derived from this script's location rather than hardcoded —
# the project directory name/path can differ across workspaces.
# Cloudera AI runs scripts cell by cell in a kernel where __file__ is undefined;
# there the working directory is the project root.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()


def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)


print("=== Installing Portfolio Optimization Dependencies ===")
print(f"Python: {sys.version}")

# Upgrade numpy first — CAI ships 1.26.4, project needs >=2.0.0.
# Must precede other installs to avoid binary incompatibility.
# Same bounds as requirements.txt: an unbounded upgrade installed pandas 3 here
# only for the next command to remove it again and install pandas 2.3.
run("pip install --upgrade 'numpy>=2.0.0,<2.5' 'pandas>=2.0,<2.4'")

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
