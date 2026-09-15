"""CML Model serving entry point for the LightGBM returns forecaster.

CML looks for a `predict.py` at the project root (with a
`predict_with_metrics` function) when building a Model from a registered
model version. This loads the trained forecaster from its saved native
file and serves per-ticker return predictions from recent price history,
mirroring `ReturnsForecaster.predict()`.
"""
import os

import pandas as pd

from portfolio_optimization.forecasting.lightgbm_model import ReturnsForecaster

MODEL_PATH = os.environ.get(
    "RETURNS_MODEL_PATH", "/home/cdsw/portfolio-optimization/models/returns_forecaster.lgb"
)

_forecaster = ReturnsForecaster()
_forecaster.load(MODEL_PATH)


def predict_with_metrics(args):
    """CML Model entry point.

    Parameters
    ----------
    args : dict
        Expected shape:
            {"prices": {"<ticker>": {"<date>": <price>, ...}, ...}}
        Recent price history per ticker, long enough for feature
        computation (same requirement as `ReturnsForecaster.predict`).

    Returns
    -------
    dict
        {"predicted_returns": {"<ticker>": <float>, ...}}
    """
    # CML calls this once with a placeholder payload during the build's
    # self-test, so a missing/empty 'prices' key must return a valid
    # response rather than raise.
    prices_arg = args.get("prices") if isinstance(args, dict) else None
    if not prices_arg:
        return {"predicted_returns": {}, "error": "missing 'prices' in request"}

    prices = pd.DataFrame(prices_arg)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    predictions = _forecaster.predict(prices)
    tickers = prices.columns.tolist()

    return {
        "predicted_returns": {
            ticker: float(value) for ticker, value in zip(tickers, predictions)
        }
    }


if __name__ == "__main__":
    # Local smoke test: python predict.py
    import json

    from portfolio_optimization.utils import get_input_data

    dataset = os.environ.get("PORTFOLIO_OPT_DATASET", "dow30")
    data_path = f"/home/cdsw/portfolio-optimization/data/stock_data/{dataset}.csv"
    sample_prices = get_input_data(data_path).tail(300)

    sample_args = {
        "prices": {
            col: {str(k): v for k, v in sample_prices[col].items()}
            for col in sample_prices.columns
        }
    }
    result = predict_with_metrics(sample_args)
    print(json.dumps(result, indent=2))
