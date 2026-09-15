import os
from pathlib import Path

from portfolio_optimization.utils import download_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset = os.environ.get("PORTFOLIO_OPT_DATASET", "dow30")
data_dir = str(PROJECT_ROOT / "data" / "stock_data")

print(f"=== Downloading stock data: {dataset} ===")
print(f"Destination: {data_dir}")

download_data(data_dir, datasets=[dataset])

print(f"=== {dataset} data downloaded ===")
