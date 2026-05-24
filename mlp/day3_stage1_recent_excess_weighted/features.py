"""
Feature entrypoint for the recent-window excess-target MLP branch.

The feature set stays aligned with the current recent-window leader. The
training target is changed from raw 3-day return to 3-day excess return versus
CSI500, while portfolio evaluation still uses raw realized returns minus the
benchmark.
"""
from __future__ import annotations

import pandas as pd

from ridge.day3_optimized.features import (  # noqa: F401
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    RAW_RETURN_COLUMN,
    build_features,
    prediction_frame,
)

TARGET_COLUMN = "target_excess_3d"


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, RAW_RETURN_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df

