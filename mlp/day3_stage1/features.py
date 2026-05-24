"""
Feature entrypoint for the stage1 3-day MLP model.
"""
from ridge.day3_optimized.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
