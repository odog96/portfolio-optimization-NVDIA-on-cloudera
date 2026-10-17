"""Deploy the Streamlit rebalancing demo as a Cloudera AI Application.

Equivalent to the `start_application` task in .project-metadata.yaml, driven
through cmlapi so it can be run from a session.

The app needs a GPU: demo/rebalancing_streamlit_app.py solves on cuOpt, and
this repo has no CPU solver fallback. Defaults therefore request a CUDA
runtime, 1 GPU, and the workspace's GPU resource group.

Usage
-----
    python scripts/deploy_app.py                  # deploy (or reuse + restart)
    python scripts/deploy_app.py --gpu-group 2    # use the L40S group instead
    python scripts/deploy_app.py --cpu-only       # no GPU (app will fail to solve)
    python scripts/deploy_app.py --status         # just report current state
"""
import argparse
import os
import sys
import time

import cmlapi
from cmlapi.rest import ApiException

APP_NAME = os.environ.get("APP_NAME", "Portfolio Rebalancing Demo")
SUBDOMAIN = os.environ.get("APP_SUBDOMAIN", "portfolio-opt")
SCRIPT = "scripts/launch_streamlit.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-group", type=int, default=1,
                   help="Resource group id for GPU. 1 = Default GPU Group (A10G), "
                        "2 = ls40 (L40S). See list_all_resource_groups().")
    p.add_argument("--cpu-only", action="store_true",
                   help="Deploy without a GPU. The app will not be able to solve.")
    p.add_argument("--cpu", type=float, default=2)
    p.add_argument("--memory", type=float, default=16)
    p.add_argument("--status", action="store_true", help="Report state and exit.")
    p.add_argument("--recreate", action="store_true",
                   help="Delete an existing app of this name and deploy fresh.")
    return p.parse_args()


def pick_runtime(client, want_gpu):
    """Newest PBJ Workbench Python 3.12 runtime, CUDA if a GPU is wanted."""
    kind = "cuda" if want_gpu else "standard"
    matches = [
        r.image_identifier
        for r in client.list_runtimes(page_size=200).runtimes
        if f"pbj-workbench-python3.12-{kind}" in r.image_identifier
        and r.status == "ENABLED"
    ]
    if not matches:
        raise RuntimeError(
            f"No enabled PBJ Workbench Python 3.12 {kind} runtime found. "
            "Set the runtime explicitly if this workspace names them differently."
        )
    return sorted(matches)[-1]


def find_app(client, project_id):
    for app in client.list_applications(project_id).applications:
        if app.name == APP_NAME:
            return app
    return None


def main():
    args = parse_args()
    project_id = os.environ.get("CDSW_PROJECT_ID")
    if not project_id:
        print("CDSW_PROJECT_ID not set — this must run inside Cloudera AI.")
        sys.exit(1)

    client = cmlapi.default_client()
    existing = find_app(client, project_id)

    if args.status:
        if existing:
            print(f"{existing.name}: {existing.status}")
            print(f"  id: {existing.id}")
            print(f"  url: https://{existing.subdomain}.{os.environ['CDSW_DOMAIN']}")
        else:
            print(f"No application named '{APP_NAME}' in this project.")
        return

    if existing and args.recreate:
        print(f"Deleting existing application {existing.id} ...")
        client.delete_application(project_id, existing.id)
        time.sleep(5)
        existing = None

    if existing:
        print(f"Application '{APP_NAME}' already exists ({existing.id}), "
              f"status {existing.status}. Restarting to pick up code changes.")
        client.restart_application(project_id, existing.id)
    else:
        want_gpu = not args.cpu_only
        runtime = pick_runtime(client, want_gpu)
        print(f"Creating application '{APP_NAME}'")
        print(f"  script:   {SCRIPT}")
        print(f"  runtime:  {runtime}")
        print(f"  gpu:      {1 if want_gpu else 0}"
              f"{f' (resource group {args.gpu_group})' if want_gpu else ''}")
        print(f"  cpu/mem:  {args.cpu} / {args.memory} GiB")

        req = cmlapi.CreateApplicationRequest(
            project_id=project_id,
            name=APP_NAME,
            subdomain=SUBDOMAIN,
            description="GPU vs CPU portfolio rebalancing comparison (Streamlit).",
            script=SCRIPT,
            cpu=args.cpu,
            memory=args.memory,
            nvidia_gpu=1 if want_gpu else 0,
            runtime_identifier=runtime,
            # The demo is read-only and public-facing in a workshop setting.
            # Set this False if the workspace requires authenticated access.
            bypass_authentication=True,
        )
        if want_gpu:
            req.resource_group_id = args.gpu_group

        try:
            app = client.create_application(req, project_id)
        except ApiException as e:
            print(f"  Create failed: {e.reason}")
            print(f"  {e.body}")
            sys.exit(1)
        existing = app
        print(f"  Created: {app.id}")

    # ------------------------------------------------------------ wait for it
    print("\nWaiting for the application to start (up to 10 min) ...")
    for i in range(60):
        app = client.get_application(project_id, existing.id)
        print(f"  [{i + 1}] {app.status}", flush=True)
        if app.status in ("running", "APPLICATION_RUNNING"):
            break
        if "fail" in str(app.status).lower() or "error" in str(app.status).lower():
            print("  Application failed to start. Check its logs in the UI.")
            sys.exit(1)
        time.sleep(10)

    url = f"https://{SUBDOMAIN}.{os.environ['CDSW_DOMAIN']}"
    print(f"\n=== {app.status} ===")
    print(f"  {url}")
    print("\nFirst load installs streamlit/plotly/squarify and may download price")
    print("data, so give it a minute before the page responds.")


if __name__ == "__main__":
    main()
