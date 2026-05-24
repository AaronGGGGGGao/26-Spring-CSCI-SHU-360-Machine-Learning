"""
Feature entrypoint for the tuned stage1 3-day MLP model on the extended 2024 dataset.

This variant intentionally keeps the exact same feature definition as the current
best model. The only intended experimental difference is the longer training
history loaded from `data_2024/`.
"""
from mlp.day3_stage1_tuned.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
