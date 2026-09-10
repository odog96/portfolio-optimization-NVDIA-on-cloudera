from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FeatureConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    momentum_windows: list[int] = Field(
        default=[5, 10, 21],
        description="Lookback windows (in trading days) for momentum features",
    )
    volatility_windows: list[int] = Field(
        default=[10, 21],
        description="Lookback windows for rolling volatility",
    )
    rsi_window: int = Field(default=14, gt=0, description="RSI lookback period")
    include_cross_asset_corr: bool = Field(
        default=True,
        description="Include rolling pairwise correlation features",
    )
    cross_asset_corr_window: int = Field(
        default=21, gt=0, description="Window for rolling cross-asset correlation"
    )


class LightGBMConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    n_estimators: int = Field(default=500, gt=0)
    max_depth: int = Field(default=6, ge=-1)
    learning_rate: float = Field(default=0.05, gt=0)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    min_child_samples: int = Field(default=20, gt=0)
    reg_alpha: float = Field(default=0.1, ge=0)
    reg_lambda: float = Field(default=0.1, ge=0)
    early_stopping_rounds: int = Field(default=50, gt=0)
    seed: int = Field(default=42)


class GARCHConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    p: int = Field(default=1, gt=0, description="GARCH lag order")
    q: int = Field(default=1, gt=0, description="ARCH lag order")
    vol_model: Literal["GARCH", "EGARCH", "GJR-GARCH"] = Field(
        default="GARCH", description="Volatility model type"
    )
    mean_model: Literal["Constant", "Zero", "AR"] = Field(
        default="Constant", description="Mean model for GARCH"
    )
    dist: Literal["normal", "t", "skewt"] = Field(
        default="t", description="Error distribution"
    )


class ForecastingConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    forecast_horizon: int = Field(
        default=5,
        gt=0,
        description="Number of trading days to forecast ahead (5=weekly)",
    )
    training_window: int = Field(
        default=756,
        gt=0,
        description="Number of trading days for training lookback (~3 years)",
    )
    retrain_frequency: int = Field(
        default=21,
        gt=0,
        description="Retrain every N trading days (~monthly)",
    )
    return_type: Literal["LOG", "LINEAR"] = Field(
        default="LOG", description="Return computation method"
    )
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    garch: GARCHConfig = Field(default_factory=GARCHConfig)
    caii_endpoint: Optional[str] = Field(
        default=None,
        description="CAII endpoint URL for returns model. None = use local model.",
    )
    caii_covariance_endpoint: Optional[str] = Field(
        default=None,
        description="CAII endpoint URL for covariance model. None = use local GARCH.",
    )
