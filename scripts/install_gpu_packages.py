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

# Verify imports in a FRESH interpreter. Importing here would fail with a
# misleading "No module named 'numba.cuda.core'": this process started before
# pip wrote those files, and Python's import finder caches directory listings,
# so newly created namespace-package directories are invisible to it.
print("Verifying GPU package imports (fresh interpreter)...")
verify = (
    "import cuml, cuopt; "
    "print(f'cuML {cuml.__version__}, cuOpt {cuopt.__version__} - OK')"
)
result = subprocess.run([sys.executable, "-c", verify], capture_output=True, text=True)
print(result.stdout.strip())
if result.returncode != 0:
    print(result.stderr.strip()[-2000:])
    raise SystemExit("GPU package verification failed — see the traceback above.")
print("=== GPU packages installed ===")
