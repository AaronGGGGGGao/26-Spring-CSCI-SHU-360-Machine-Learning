"""
Feature entrypoint for the recent-window weight-optimized robust-split branch.
"""
from mlp.day3_stage1_recent_window_weight_optimized.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
