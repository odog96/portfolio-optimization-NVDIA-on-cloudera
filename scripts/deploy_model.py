"""Deploy the registered returns forecaster to a CML model endpoint.

Follows the cmlapi pattern from CAI-baseline-workshop/module1/04_deploy.py:
  1. Read registration info from train_and_register.py
  2. Create CML model with registered_model_id
  3. Build with runtime
  4. Wait for build, then deploy
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/cdsw/portfolio-optimization")

MODEL_NAME = os.environ.get("CAII_MODEL_NAME", "PortfolioReturnsForecaster")

print(f"=== Deploying {MODEL_NAME} ===")

# ---------------------------------------------------------------------------
# Step 1: Load registration info
# ---------------------------------------------------------------------------
project_id = os.environ.get("CDSW_PROJECT_ID")
if not project_id:
    print("CDSW_PROJECT_ID not set — not running in CML.")
    sys.exit(1)

reg_info_path = "outputs/registration_info.json"
if os.path.exists(reg_info_path):
    with open(reg_info_path) as f:
        reg_info = json.load(f)
    registered_model_id = reg_info["registered_model_id"]
    model_version_id = reg_info["model_version_id"]
    print(f"  Registered model: {registered_model_id}")
    print(f"  Version: {model_version_id}")
else:
    print(f"  {reg_info_path} not found.")
    print("  Run scripts/train_and_register.py first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: Create CML model
# ---------------------------------------------------------------------------
import cmlapi
from cmlapi.rest import ApiException

cml_client = cmlapi.default_client()

print("\n--- Creating CML model ---")

create_model_req = cmlapi.CreateModelRequest(
    project_id=project_id,
    name=MODEL_NAME,
    description="LightGBM returns forecaster for portfolio optimization",
    registered_model_id=registered_model_id,
    disable_authentication=True,
)

try:
    cml_model = cml_client.create_model(
        body=create_model_req, project_id=project_id
    )
    print(f"  Model created: {cml_model.id}")
except ApiException as e:
    if "already has a model with that name" in str(e.body):
        print(f"  Model '{MODEL_NAME}' already exists, reusing...")
        models = cml_client.list_models(project_id)
        cml_model = next(
            (m for m in models.models if m.name == MODEL_NAME), None
        )
        if not cml_model:
            print("  Could not find existing model.")
            sys.exit(1)
        print(f"  Using existing: {cml_model.id}")
    else:
        print(f"  Error: {e.reason}")
        print(f"  {e.body}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3: Create model build
# ---------------------------------------------------------------------------
print("\n--- Creating model build ---")

# Auto-detect a Python 3.12 standard runtime, or use env override
runtime_id = os.environ.get("CAII_RUNTIME")
if not runtime_id:
    runtimes = cml_client.list_runtimes(page_size=200)
    for r in runtimes.runtimes:
        img = getattr(r, "image_identifier", "")
        if "python3.12-standard" in img and "workbench" in img:
            runtime_id = img
            break
    if not runtime_id:
        print("  Could not find a Python 3.12 standard workbench runtime.")
        print("  Set CAII_RUNTIME env var to the correct runtime identifier.")
        sys.exit(1)
print(f"  Runtime: {runtime_id}")

create_build_req = cmlapi.CreateModelBuildRequest(
    registered_model_version_id=str(model_version_id),
    runtime_identifier=runtime_id,
    comment="Portfolio returns forecaster",
)

try:
    build = cml_client.create_model_build(
        body=create_build_req,
        project_id=project_id,
        model_id=cml_model.id,
    )
    print(f"  Build started: {build.id}")
except ApiException as e:
    print(f"  Build error: {e.reason}")
    print(f"  {e.body}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: Wait for build
# ---------------------------------------------------------------------------
print("\n--- Waiting for build (up to 15 min) ---")

max_checks = 30
for i in range(max_checks):
    build_status = cml_client.get_model_build(
        project_id=project_id,
        model_id=cml_model.id,
        build_id=build.id,
    )
    status = build_status.status
    print(f"  [{i+1}/{max_checks}] {status}")

    if status == "built":
        print("  Build complete.")
        break
    elif status == "build failed":
        print("  Build failed. Check CML UI for logs.")
        sys.exit(1)
    else:
        time.sleep(30)
else:
    print("  Build did not complete in time. Deploy manually when ready.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Step 5: Deploy
# ---------------------------------------------------------------------------
print("\n--- Deploying ---")

deploy_req = cmlapi.CreateModelDeploymentRequest(cpu="2", memory="4")

try:
    deployment = cml_client.create_model_deployment(
        body=deploy_req,
        project_id=project_id,
        model_id=cml_model.id,
        build_id=build.id,
    )
    print(f"  Deployment created: {deployment.id}")
    print(f"\n=== Deployment complete ===")
    print(f"  Model: {MODEL_NAME}")
    print(f"  CML Model ID: {cml_model.id}")
    print(f"  Build ID: {build.id}")
    print(f"  Deployment ID: {deployment.id}")
    print(f"\n  Access: Models > {MODEL_NAME} > Deployments")
except ApiException as e:
    print(f"  Deploy error: {e.reason}")
    print(f"  {e.body}")
    print(f"\n  Deploy manually: Models > {MODEL_NAME} > Builds > Deploy")
