"""Cloudera AI Application launcher: starts the Streamlit rebalancing demo."""

import os
import subprocess
import sys
import time
from pathlib import Path

# Cloudera AI runs scripts cell by cell in a kernel where __file__ is undefined;
# there the working directory is the project root.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

app_path = PROJECT_ROOT / "demo" / "rebalancing_streamlit_app.py"
if not app_path.is_file():
    # Fail with the path rather than leaving the Application stuck in Starting.
    raise SystemExit(f"Streamlit app not found at {app_path}")

# Install the demo dependencies, plus pydantic so pip fills in its dependencies
# for this runtime. All Python 3.12 runtimes share ~/.local, and pydantic
# installed from another runtime can be missing typing_inspection, which that
# runtime had built in. Not the full requirements.txt: its mlflow pin would
# downgrade pyarrow below what the GPU libraries need on every start.
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install",
        "-r", str(PROJECT_ROOT / "demo" / "requirements.txt"),
        "pydantic>=2.12.5",
    ]
)

# Streamlit inherits this process's working directory.
os.chdir(PROJECT_ROOT)

# CDSW_APP_PORT is set by Cloudera AI for Applications.
port = os.environ.get("CDSW_APP_PORT", "8090")

# Idle shutdown: the app touches the heartbeat file on every user interaction.
# After IDLE_SHUTDOWN_MINUTES without one (default 60, 0 disables), Streamlit is
# stopped and this script exits, which ends the Application and frees its GPU.
idle_minutes = float(os.environ.get("IDLE_SHUTDOWN_MINUTES", "60"))
heartbeat = PROJECT_ROOT / "outputs" / f".app_heartbeat_{os.environ.get('CDSW_ENGINE_ID', port)}"
heartbeat.parent.mkdir(exist_ok=True)
heartbeat.touch()

# --server.headless=true is required: without it Streamlit's first start asks
# for an email address on the console and waits for an answer that never comes
# inside an Application, so the Application never leaves Starting.
proc = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
    ],
    # If the app crashes in native code (e.g. a GPU library segfault), print
    # the Python stack of every thread to the Application log.
    env={**os.environ, "PYTHONFAULTHANDLER": "1", "APP_HEARTBEAT_FILE": str(heartbeat)},
)

while proc.poll() is None:
    time.sleep(60)
    idle = (time.time() - heartbeat.stat().st_mtime) / 60
    if idle_minutes > 0 and idle >= idle_minutes:
        print(f"No activity for {idle:.0f} minutes; stopping the Application to free its GPU.", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        heartbeat.unlink(missing_ok=True)
        # Ask Cloudera AI to stop this Application. The Application record does
        # not name its container, so act only if exactly one running
        # Application uses this launcher.
        try:
            import cmlapi

            client = cmlapi.default_client()
            project_id = os.environ["CDSW_PROJECT_ID"]
            running = [
                a for a in client.list_applications(project_id).applications
                if a.status == "APPLICATION_RUNNING" and a.script == "scripts/launch_streamlit.py"
            ]
            if len(running) == 1:
                client.stop_application(project_id, running[0].id)
        except Exception as exc:
            print(f"Could not stop the Application through the API: {exc}", flush=True)
        # The launcher may run inside a kernel where sys.exit does not end the
        # process; exit hard so the Application ends either way.
        os._exit(0)

heartbeat.unlink(missing_ok=True)
if proc.returncode:
    raise subprocess.CalledProcessError(proc.returncode, proc.args)
