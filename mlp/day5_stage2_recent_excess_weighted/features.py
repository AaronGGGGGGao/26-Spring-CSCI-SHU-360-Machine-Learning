"""
Feature entrypoint for the Stage 2 5-day excess-target MLP branch.

The model trains on stock 5-day excess return versus CSI500. Portfolio
evaluation still uses realized raw stock 5-day return minus benchmark 5-day
return.
"""
from __future__ import annotations

import pandas as pd

from mlp.day5_stage2_recent_window.features import (  # noqa: F401
    EXCESS_TARGET_COLUMN,
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    RAW_RETURN_COLUMN,
    build_features,
    prediction_frame,
)

TARGET_COLUMN = EXCESS_TARGET_COLUMN


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, RAW_RETURN_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df
