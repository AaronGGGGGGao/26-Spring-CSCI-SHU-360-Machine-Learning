"""
Feature entrypoint for the robust-split tuned stage1 MLP model.

This variant intentionally keeps the same feature engineering as the current
stage1 winner. The only experimental change is the self-test split design.
"""
from mlp.day3_stage1_tuned.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
