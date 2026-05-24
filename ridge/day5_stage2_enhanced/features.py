"""Feature entrypoint for the Stage 2 5-day enhanced Ridge branch."""
from mlp.day5_stage2_recent_window.features import (  # noqa: F401
    EXCESS_TARGET_COLUMN,
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    RAW_RETURN_COLUMN,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
