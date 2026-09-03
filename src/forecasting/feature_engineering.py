import numpy as np
import pandas as pd

from portfolio_optimization.forecasting.config import FeatureConfig


def compute_features(
    prices: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute per-asset technical features from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns are asset tickers, index is DatetimeIndex, values are prices.
    config : FeatureConfig, optional
        Feature computation parameters. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (ticker, feature_name). Rows aligned with prices
        index. NaN rows at the start (due to rolling windows) are dropped.
    """
    if config is None:
        config = FeatureConfig()

    log_returns = np.log(prices / prices.shift(1))
    tickers = prices.columns.tolist()
    frames = []

    for ticker in tickers:
        ret = log_returns[ticker]
        px = prices[ticker]
        feats = {}

        # Momentum: cumulative return over window
        for w in config.momentum_windows:
            feats[f"momentum_{w}d"] = ret.rolling(w).sum()

        # Rolling volatility
        for w in config.volatility_windows:
            feats[f"volatility_{w}d"] = ret.rolling(w).std()

        # RSI
        feats[f"rsi_{config.rsi_window}d"] = _compute_rsi(px, config.rsi_window)

        # Rolling mean return
        feats["mean_return_21d"] = ret.rolling(21).mean()

        # Price relative to moving averages
        feats["price_vs_sma21"] = px / px.rolling(21).mean() - 1
        feats["price_vs_sma63"] = px / px.rolling(63).mean() - 1

        # Volume proxy: rolling return dispersion (high dispersion = high activity)
        feats["return_dispersion_10d"] = ret.rolling(10).apply(
            lambda x: x.max() - x.min(), raw=True
        )

        ticker_df = pd.DataFrame(feats, index=prices.index)
        ticker_df.columns = pd.MultiIndex.from_product(
            [[ticker], ticker_df.columns]
        )
        frames.append(ticker_df)

    result = pd.concat(frames, axis=1)

    # Cross-asset rolling correlation (mean pairwise correlation for each asset)
    if config.include_cross_asset_corr and len(tickers) > 1:
        corr_features = _compute_cross_asset_correlation(
            log_returns, tickers, config.cross_asset_corr_window
        )
        result = pd.concat([result, corr_features], axis=1)

    min_window = max(
        max(config.momentum_windows),
        max(config.volatility_windows),
        config.rsi_window,
        63,  # longest SMA
    )
    result = result.iloc[min_window:]
    return result


def build_training_data(
    prices: pd.DataFrame,
    forecast_horizon: int = 5,
    return_type: str = "LOG",
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature matrix X and target matrix y for supervised learning.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data.
    forecast_horizon : int
        Number of days ahead to predict returns for.
    return_type : str
        "LOG" or "LINEAR".
    config : FeatureConfig, optional
        Feature computation parameters.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix. MultiIndex columns (ticker, feature_name).
    y : pd.DataFrame
        Target returns over forecast_horizon. Columns are tickers.
    """
    features = compute_features(prices, config)

    if return_type == "LOG":
        forward_returns = np.log(prices.shift(-forecast_horizon) / prices)
    else:
        forward_returns = (prices.shift(-forecast_horizon) - prices) / prices

    # Align features and targets, drop NaN rows
    common_idx = features.index.intersection(forward_returns.dropna().index)
    X = features.loc[common_idx]
    y = forward_returns.loc[common_idx]

    return X, y


def flatten_features_for_training(
    X: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Flatten multi-asset feature/target into a single model's training set.

    Stacks all assets into rows so one model learns across all assets.
    Adds the ticker as a categorical feature.

    Parameters
    ----------
    X : pd.DataFrame
        MultiIndex columns (ticker, feature_name) from build_training_data.
    y : pd.DataFrame
        Columns are tickers, one target per asset per date.

    Returns
    -------
    X_flat : pd.DataFrame
        One row per (date, asset). Regular column index.
    y_flat : pd.Series
        Corresponding target returns.
    """
    tickers = y.columns.tolist()
    rows_X = []
    rows_y = []

    for ticker in tickers:
        if ticker not in X.columns.get_level_values(0):
            continue
        ticker_features = X[ticker].copy()
        ticker_features["ticker"] = ticker
        rows_X.append(ticker_features)
        rows_y.append(y[ticker])

    X_flat = pd.concat(rows_X, axis=0)
    y_flat = pd.concat(rows_y, axis=0)

    X_flat["ticker"] = X_flat["ticker"].astype("category")
    return X_flat, y_flat


def _compute_rsi(prices: pd.Series, window: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_cross_asset_correlation(
    returns: pd.DataFrame, tickers: list[str], window: int
) -> pd.DataFrame:
    """Mean rolling pairwise correlation for each asset vs. all others."""
    frames = {}
    rolling_corr = returns.rolling(window).corr()

    for ticker in tickers:
        try:
            ticker_corr = rolling_corr.loc[(slice(None), ticker), :]
            ticker_corr = ticker_corr.drop(columns=ticker, errors="ignore")
            mean_corr = ticker_corr.mean(axis=1)
            mean_corr.index = mean_corr.index.droplevel(1)
            frames[(ticker, "mean_cross_corr")] = mean_corr
        except (KeyError, TypeError):
            frames[(ticker, "mean_cross_corr")] = pd.Series(
                np.nan, index=returns.index
            )

    result = pd.DataFrame(frames)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result
