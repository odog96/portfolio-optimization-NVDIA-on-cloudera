"""Deploy the registered returns forecaster model to CAII.

AMP task: runs after train_and_register.py. Deploys the latest version of
PortfolioReturnsForecaster from MLflow model registry to Cloudera AI
Inference Services (CAII).

CAII deployment is done via the UI in most setups. This script retrieves
the model artifact path and prints instructions. Programmatic deployment
via cmlapi is included when available.
"""
import os
import sys

sys.path.insert(0, "/home/cdsw/portfolio-optimization")

MODEL_NAME = os.environ.get(
    "CAII_MODEL_NAME", "PortfolioReturnsForecaster"
)

print(f"=== Deploying {MODEL_NAME} to CAII ===")

# Step 1: Get latest model version from MLflow
import mlflow

client = mlflow.tracking.MlflowClient()

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(f"No versions found for model '{MODEL_NAME}'.")
        print("Run scripts/train_and_register.py first.")
        sys.exit(1)

    latest = max(versions, key=lambda v: int(v.version))
    print(f"  Model: {MODEL_NAME}")
    print(f"  Version: {latest.version}")
    print(f"  Source: {latest.source}")
    print(f"  Status: {latest.status}")
except Exception as e:
    print(f"Error querying MLflow: {e}")
    sys.exit(1)

# Step 2: Attempt programmatic deployment via cmlapi
deployed = False
try:
    import cmlapi

    cml_client = cmlapi.default_client()
    project_id = os.environ.get("CDSW_PROJECT_ID")

    if project_id:
        print(f"\n--- Creating CAII model endpoint ---")
        print(f"  Project ID: {project_id}")

        model_body = cmlapi.CreateModelRequest(
            name=MODEL_NAME,
            description="LightGBM returns forecaster for portfolio optimization",
            project_id=project_id,
        )
        model = cml_client.create_model(model_body, project_id)
        print(f"  CAII model created: {model.id}")

        build_body = cmlapi.CreateModelBuildRequest(
            model_id=model.id,
            project_id=project_id,
            registered_model_version_id=latest.version,
        )
        build = cml_client.create_model_build(build_body, project_id, model.id)
        print(f"  Build started: {build.id}")

        deploy_body = cmlapi.CreateModelDeploymentRequest(
            model_id=model.id,
            build_id=build.id,
            project_id=project_id,
            cpu=1,
            memory=4,
            gpu=0,
        )
        deployment = cml_client.create_model_deployment(
            deploy_body, project_id, model.id, build.id
        )
        print(f"  Deployment started: {deployment.id}")
        deployed = True
    else:
        print("\n  CDSW_PROJECT_ID not set — skipping programmatic deployment.")

except ImportError:
    print("\n  cmlapi not available — skipping programmatic deployment.")
except Exception as e:
    print(f"\n  Programmatic deployment failed: {e}")

# Step 3: Print manual deployment instructions if needed
if not deployed:
    print("\n--- Manual CAII Deployment Instructions ---")
    print(f"  1. Open Cloudera AI → Model Registry")
    print(f"  2. Find '{MODEL_NAME}' (version {latest.version})")
    print(f"  3. Click 'Deploy' → select CAII endpoint")
    print(f"  4. Set resources: CPU=1, Memory=4GB, GPU=0")
    print(f"  5. Deploy and note the endpoint URL")
    print(f"  6. Set CAII_ENDPOINT env var to the endpoint URL")
    print(f"  7. Set CAII_API_KEY env var to your API key")

print("\n=== Deployment script complete ===")
