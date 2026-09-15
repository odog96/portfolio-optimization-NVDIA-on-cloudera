"""CML Model serving entry point for the LightGBM returns forecaster.

Conforms to the Cloudera AI Workbench "Models" convention: a script file
(this one) plus a function, decorated with `@cdsw.model_metrics`, that CML
calls once per inference request. The function receives the parsed JSON
request body as a dict and must return a JSON-serializable value.

`cdsw` is provided by the CML Model runtime container itself — it is not
pip-installable and is unavailable in a plain dev session. When it's
missing (i.e. we're smoke-testing locally, not running inside a deployed
Model), we fall back to a no-op decorator so this file still runs and can
be exercised end-to-end before ever spending a real build/deploy cycle.
"""
import os

import pandas as pd

from portfolio_optimization.forecasting.lightgbm_model import ReturnsForecaster

try:
    import cdsw

    _HAVE_CDSW = True
except ImportError:
    _HAVE_CDSW = False

    class _NoOpCdsw:
        """Local-only stand-in for the real `cdsw` module, used purely so
        this file can be smoke-tested outside a CML Model container. Never
        used in production — real deployments always have `cdsw`."""

        @staticmethod
        def model_metrics(func):
            return func

        @staticmethod
        def track_metric(key, value):
            print(f"[local test] track_metric({key!r}, {value!r})")

    cdsw = _NoOpCdsw()

MODEL_PATH = os.environ.get(
    "RETURNS_MODEL_PATH",
    "/home/cdsw/portfolio-optimization/models/returns_forecaster.lgb",
)

_forecaster = ReturnsForecaster()
_forecaster.load(MODEL_PATH)


@cdsw.model_metrics
def predict_with_metrics(args):
    """CML Model entry point.

    Parameters
    ----------
    args : dict
        Expected shape:
            {"prices": {"<ticker>": {"<date>": <price>, ...}, ...}}
        Recent price history per ticker, long enough for feature
        computation (same requirement as `ReturnsForecaster.predict`).

        CML calls this once with a placeholder/empty payload during a
        build's self-test, so a missing or malformed `prices` key must
        return a valid response rather than raise.

    Returns
    -------
    dict
        {"predicted_returns": {"<ticker>": <float>, ...}}
        or, on bad/missing input: {"predicted_returns": {}, "error": "..."}
    """
    prices_arg = args.get("prices") if isinstance(args, dict) else None
    if not prices_arg:
        return {"predicted_returns": {}, "error": "missing 'prices' in request"}

    try:
        prices = pd.DataFrame(prices_arg)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        predictions = _forecaster.predict(prices)
        tickers = prices.columns.tolist()
    except Exception as e:
        return {"predicted_returns": {}, "error": f"{type(e).__name__}: {e}"}

    predicted_returns = {
        ticker: float(value) for ticker, value in zip(tickers, predictions)
    }

    cdsw.track_metric("n_tickers", len(predicted_returns))
    if predicted_returns:
        cdsw.track_metric(
            "mean_predicted_return",
            sum(predicted_returns.values()) / len(predicted_returns),
        )

    return {"predicted_returns": predicted_returns}


if __name__ == "__main__":
    # Local smoke test: python serve.py
    import json

    from portfolio_optimization.utils import get_input_data

    print(f"cdsw available: {_HAVE_CDSW}")

    dataset = os.environ.get("PORTFOLIO_OPT_DATASET", "dow30")
    data_path = f"/home/cdsw/portfolio-optimization/data/stock_data/{dataset}.csv"
    sample_prices = get_input_data(data_path).tail(300)

    print("\n--- Test 1: empty/placeholder payload (build self-test) ---")
    print(json.dumps(predict_with_metrics({}), indent=2))

    print("\n--- Test 2: real payload ---")
    sample_args = {
        "prices": {
            col: {str(k): v for k, v in sample_prices[col].items()}
            for col in sample_prices.columns
        }
    }
    result = predict_with_metrics(sample_args)
    print(json.dumps(result, indent=2))

    print("\n--- Test 3: malformed payload ---")
    print(json.dumps(predict_with_metrics({"prices": "not-a-dict-of-dicts"}), indent=2))
