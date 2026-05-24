"""
Feature entrypoint for the robust-split style-dynamic MLP.
"""
from mlp.day3_stage1_style_dynamic.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)

