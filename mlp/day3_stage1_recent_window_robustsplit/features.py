"""
Feature entrypoint for the robust-split recent-window tuned MLP.
"""
from mlp.day3_stage1_recent_window.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)

