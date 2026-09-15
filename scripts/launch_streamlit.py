import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)


# Install Streamlit demo dependencies
run("pip install streamlit==1.58.0 plotly==6.8.0 squarify==0.4.4")

# Launch the app — CDSW_APP_PORT is set by CAI for applications
import os

port = os.environ.get("CDSW_APP_PORT", "8090")
app_path = PROJECT_ROOT / "demo" / "rebalancing_streamlit_app.py"

run(
    f"streamlit run {app_path} "
    f"--server.port={port} "
    f"--server.address=127.0.0.1 "
    f"--server.enableCORS=false "
    f"--server.enableXsrfProtection=false"
)
