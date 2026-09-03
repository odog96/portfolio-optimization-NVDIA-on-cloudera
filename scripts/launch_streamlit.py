import subprocess
import sys


def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)


# Install Streamlit demo dependencies
run("pip install streamlit==1.58.0 plotly==6.8.0 squarify==0.4.4")

# Launch the app — CDSW_APP_PORT is set by CAI for applications
import os

port = os.environ.get("CDSW_APP_PORT", "8090")

run(
    f"streamlit run /home/cdsw/portfolio-optimization/demo/rebalancing_streamlit_app.py "
    f"--server.port={port} "
    f"--server.address=127.0.0.1 "
    f"--server.enableCORS=false "
    f"--server.enableXsrfProtection=false"
)
