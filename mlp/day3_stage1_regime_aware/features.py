"""
Regime-aware feature entrypoint for the stage1 3-day tuned MLP.

This keeps the existing tuned-MLP feature panel and adds a small set of market
state features so the model can condition stock selection on the current market
environment instead of learning one static mapping across all regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlp.day3_stage1_tuned.features import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features as base_build_features,
)

REGIME_FEATURE_COLUMNS = [
    "mkt_ret_3d",
    "mkt_ret_10d",
    "mkt_ret_20d",
    "mkt_vol_5d",
    "mkt_vol_10d",
    "mkt_trend_5_20",
    "mkt_drawdown_20d",
    "mkt_range_5d",
    "mkt_breadth_5d",
    "mkt_breadth_10d",
    "mkt_up_ratio_10d",
    "mkt_vol_regime",
    "ret3_x_mkt_trend",
    "beta_x_mkt_vol",
    "excess3_x_mkt_breadth",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS


def build_features(prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    panel = base_build_features(prices, index_df).copy()

    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    idx_close = idx["close"].astype(float)
    idx["mkt_ret_3d"] = idx_close.pct_change(3)
    idx["mkt_ret_10d"] = idx_close.pct_change(10)
    idx["mkt_ret_20d"] = idx_close.pct_change(20)
    idx["mkt_ret_1d"] = idx_close.pct_change(1)
    idx["mkt_vol_5d"] = idx["mkt_ret_1d"].rolling(5).std()
    idx["mkt_vol_10d"] = idx["mkt_ret_1d"].rolling(10).std()
    idx["mkt_trend_5_20"] = idx_close.rolling(5).mean() / idx_close.rolling(20).mean() - 1.0
    idx["mkt_drawdown_20d"] = idx_close / idx_close.rolling(20).max() - 1.0
    idx["mkt_range_5d"] = (idx["high"].rolling(5).max() / idx["low"].rolling(5).min()) - 1.0
    vol_med = idx["mkt_vol_10d"].rolling(60).median()
    idx["mkt_vol_regime"] = (idx["mkt_vol_10d"] > vol_med).astype(float)

    breadth = panel.groupby("date")["ret_1d"].agg(
        mkt_breadth_1d=lambda s: (s > 0).mean()
    ).reset_index()
    breadth["mkt_breadth_5d"] = breadth["mkt_breadth_1d"].rolling(5).mean()
    breadth["mkt_breadth_10d"] = breadth["mkt_breadth_1d"].rolling(10).mean()
    breadth["mkt_up_ratio_10d"] = breadth["mkt_breadth_1d"].rolling(10).sum() / 10.0

    panel = panel.merge(
        idx[
            [
                "date",
                "mkt_ret_3d",
                "mkt_ret_10d",
                "mkt_ret_20d",
                "mkt_vol_5d",
                "mkt_vol_10d",
                "mkt_trend_5_20",
                "mkt_drawdown_20d",
                "mkt_range_5d",
                "mkt_vol_regime",
            ]
        ],
        on="date",
        how="left",
    )
    panel = panel.merge(
        breadth[["date", "mkt_breadth_5d", "mkt_breadth_10d", "mkt_up_ratio_10d"]],
        on="date",
        how="left",
    )

    panel["ret3_x_mkt_trend"] = panel["ret_3d"] * panel["mkt_trend_5_20"]
    panel["beta_x_mkt_vol"] = panel["beta_20d"] * panel["mkt_vol_10d"]
    panel["excess3_x_mkt_breadth"] = panel["excess_ret_3d"] * panel["mkt_breadth_10d"]
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    return panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
