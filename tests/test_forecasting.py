import numpy as np
import pytest

from portfolio_optimization.cvar_data import CvarData
from portfolio_optimization.forecasting.scenario_adjustment import rescale_scenarios


@pytest.fixture
def returns_dict():
    """3 correlated, fat-tailed assets x 5,000 scenarios, as generate_cvar_data lays them out."""
    rng = np.random.default_rng(0)
    corr = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]])
    z = rng.standard_t(df=4, size=(5000, 3)) @ np.linalg.cholesky(corr).T
    R = (0.0005 + 0.02 * z).T  # shape (n_assets, n_scenarios)
    p = np.full(R.shape[1], 1 / R.shape[1])
    mean = R.mean(axis=1)
    return {"mean": mean, "tickers": ["A", "B", "C"], "cvar_data": CvarData(mean=mean, R=R, p=p)}


FORECAST_MEAN = np.array([0.001, -0.0005, 0.0002])
FORECAST_VOL = np.array([0.015, 0.03, 0.01])


def test_scenarios_take_forecast_mean_and_volatility(returns_dict):
    R_new = rescale_scenarios(returns_dict, FORECAST_MEAN, FORECAST_VOL)["cvar_data"].R
    np.testing.assert_allclose(R_new.mean(axis=1), FORECAST_MEAN, atol=1e-12)
    np.testing.assert_allclose(R_new.std(axis=1), FORECAST_VOL, rtol=1e-12)


def test_correlation_and_shape_are_preserved(returns_dict):
    R = returns_dict["cvar_data"].R
    R_new = rescale_scenarios(returns_dict, FORECAST_MEAN, FORECAST_VOL)["cvar_data"].R
    np.testing.assert_allclose(np.corrcoef(R_new), np.corrcoef(R), atol=1e-12)
    # Each asset's scenarios are only shifted and stretched, so their
    # standardized values (and hence skew and tail shape) are unchanged.
    standardize = lambda x: (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    np.testing.assert_allclose(standardize(R_new), standardize(R), atol=1e-10)


def test_optimizer_inputs_updated_and_input_untouched(returns_dict):
    R_before = returns_dict["cvar_data"].R.copy()
    mean_before = returns_dict["mean"].copy()
    updated = rescale_scenarios(returns_dict, FORECAST_MEAN, FORECAST_VOL)

    np.testing.assert_array_equal(updated["mean"], FORECAST_MEAN)
    np.testing.assert_array_equal(updated["cvar_data"].mean, FORECAST_MEAN)
    np.testing.assert_array_equal(updated["cvar_data"].p, returns_dict["cvar_data"].p)
    np.testing.assert_array_equal(returns_dict["cvar_data"].R, R_before)
    np.testing.assert_array_equal(returns_dict["mean"], mean_before)


@pytest.mark.parametrize(
    "mean, vol",
    [
        (FORECAST_MEAN[:2], FORECAST_VOL),  # wrong number of assets
        (FORECAST_MEAN, FORECAST_VOL[:2]),
        (FORECAST_MEAN, np.array([0.01, 0.0, 0.01])),  # zero volatility
        (FORECAST_MEAN, np.array([0.01, np.nan, 0.01])),
    ],
)
def test_invalid_forecasts_are_rejected(returns_dict, mean, vol):
    with pytest.raises(ValueError):
        rescale_scenarios(returns_dict, mean, vol)



# --- GARCH conditioning and the walk-forward estimate provider -------------

import pandas as pd

from portfolio_optimization.forecasting.config import ForecastingConfig
from portfolio_optimization.forecasting.estimate_provider import ForecastEstimateProvider
from portfolio_optimization.forecasting.garch_model import CovarianceForecaster


@pytest.fixture(scope="module")
def prices():
    """4 assets x 1,200 business days of GARCH(1,1) returns (volatility clusters)."""
    rng = np.random.default_rng(1)
    omega, alpha, beta = 2e-6, 0.10, 0.85
    r = np.zeros((1200, 4))
    var = np.full(4, omega / (1 - alpha - beta))
    for t in range(1200):
        r[t] = np.sqrt(var) * rng.standard_normal(4)
        var = omega + alpha * r[t] ** 2 + beta * var
    idx = pd.bdate_range("2018-01-01", periods=1200)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=list("ABCD"))


def test_garch_forecast_uses_the_prices_passed_in(prices):
    model = CovarianceForecaster(ForecastingConfig())
    model.train(prices.iloc[:900].tail(500))
    before = model.predict_volatility(prices.iloc[:900].tail(500))
    # Same fitted parameters, a window ending in a week of 6% daily moves:
    # the forecast must rise. Before the fix it ignored the prices passed in.
    shock = prices.iloc[:900].copy()
    for i in range(5):
        shock.loc[shock.index[-1] + pd.offsets.BDay()] = shock.iloc[-1] * (1.06 if i % 2 else 0.94)
    after = model.predict_volatility(shock.tail(500))
    assert np.all(after > 1.5 * before)


def test_provider_trains_without_look_ahead(prices):
    start = prices.index[900]
    provider = ForecastEstimateProvider(retrain_months=3).fit_schedule(
        prices, start, prices.index[1150]
    )
    dates = [d for d, _, _ in provider.models]
    assert dates[0] == start and len(dates) == 4
    for date, _, vol_model in provider.models:
        fitted = vol_model.garch_models["A"].model._y_original.index
        assert fitted.max() < date
    # A rebalance uses the latest model trained on or before its date.
    assert provider.models_for(dates[1] + pd.Timedelta(days=1))[0] == dates[1]
    with pytest.raises(ValueError):
        provider.models_for(start - pd.Timedelta(days=1))
