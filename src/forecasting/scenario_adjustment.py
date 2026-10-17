import numpy as np

from portfolio_optimization.cvar_data import CvarData


def rescale_scenarios(
    returns_dict: dict, mean: np.ndarray, volatility: np.ndarray
) -> dict:
    """Re-center and re-scale each asset's scenarios to forecast values.

    For every asset i and scenario s:

        R_new[i, s] = mean[i] + volatility[i] * (R[i, s] - m_i) / s_i

    where m_i and s_i are the mean and standard deviation of asset i's own
    scenarios. Each asset's scenarios then have exactly the forecast mean and
    volatility, while keeping their historical shape (fat tails, skew) and
    the correlation between assets, since each row is only shifted and
    stretched.

    Call after cvar_utils.generate_cvar_data() and before building the
    optimizer, so the CVaR term and the expected-return term both use the
    forecasts.

    Parameters
    ----------
    returns_dict : dict
        Must contain "cvar_data" (a CvarData whose R has shape
        (n_assets, n_scenarios)). Not modified.
    mean : np.ndarray
        Forecast expected return per asset, shape (n_assets,), in the same
        return type (log or linear) and period as the scenarios.
    volatility : np.ndarray
        Forecast volatility per asset as a decimal, shape (n_assets,),
        e.g. CovarianceForecaster.predict_volatility().

    Returns
    -------
    dict
        Shallow copy of returns_dict with "mean" set to `mean` and
        "cvar_data" replaced by the rescaled scenarios.
    """
    cvar_data = returns_dict["cvar_data"]
    R = np.asarray(cvar_data.R, dtype=float)
    mean = np.asarray(mean, dtype=float)
    volatility = np.asarray(volatility, dtype=float)

    n_assets = R.shape[0]
    if mean.shape != (n_assets,) or volatility.shape != (n_assets,):
        raise ValueError(
            f"mean and volatility must have shape ({n_assets},) to match the "
            f"{n_assets} assets in the scenarios; got {mean.shape} and "
            f"{volatility.shape}."
        )
    if not np.all(np.isfinite(volatility)) or np.any(volatility <= 0):
        raise ValueError("volatility must be finite and positive for every asset.")

    scenario_mean = R.mean(axis=1, keepdims=True)
    scenario_std = R.std(axis=1, keepdims=True)
    if np.any(scenario_std == 0):
        raise ValueError("an asset has identical returns in every scenario.")

    # Standardize each asset's scenarios, then apply the forecast mean and vol.
    R_new = mean[:, None] + volatility[:, None] * (R - scenario_mean) / scenario_std

    updated = returns_dict.copy()
    updated["mean"] = mean
    updated["cvar_data"] = CvarData(mean=mean, R=R_new, p=cvar_data.p)
    return updated

