import os

from portfolio_optimization.utils import download_data

dataset = os.environ.get("PORTFOLIO_OPT_DATASET", "dow30")
data_dir = "/home/cdsw/portfolio-optimization/data/stock_data"

print(f"=== Downloading stock data: {dataset} ===")
print(f"Destination: {data_dir}")

download_data(data_dir, datasets=[dataset])

print(f"=== {dataset} data downloaded ===")
