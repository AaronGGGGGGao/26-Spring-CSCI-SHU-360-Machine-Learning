"""
Feature entrypoint for the optimized-portfolio stage1 MLP model.

This variant intentionally keeps the same feature set as the current tuned MLP
leader. The only change is the mapping from model score to portfolio weights.
"""
from mlp.day3_stage1_tuned.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
