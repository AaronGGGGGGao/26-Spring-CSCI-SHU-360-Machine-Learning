"""Feature entrypoint for the Stage 2 5-day style-dynamic robust split."""
from mlp.day5_stage2_style_dynamic.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
