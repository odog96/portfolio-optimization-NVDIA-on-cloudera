import os
import subprocess
import sys


def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)


print("=== Installing GPU Packages (cuOpt + cuML) ===")

gpu_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "void")
if gpu_visible == "void":
    print("WARNING: No GPU detected (NVIDIA_VISIBLE_DEVICES=void).")
    print("This script should run in a GPU session. Attempting install anyway...")

run(
    "pip install --extra-index-url https://pypi.nvidia.com "
    "'cuml-cu12==26.6.*' "
    "'cuopt-cu12==26.6.*'"
)

# Verify imports
print("Verifying GPU package imports...")
import cuml
import cuopt

print(f"cuML {cuml.__version__}, cuOpt {cuopt.__version__} — OK")
print("=== GPU packages installed ===")
