"""
Stage1-oriented feature module for 3-day XGBoost.

This reuses the richer public-price feature set that was built for the
optimized 3-day Ridge experiments:
  - short-horizon returns and relative strength
  - intraday/range behavior
  - liquidity/activity z-scores
  - beta / idiosyncratic-risk features
  - cross-sectional ranks

The target remains raw 3-day forward return, not excess return.
"""
from ridge.day3_optimized.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    RAW_RETURN_COLUMN,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
