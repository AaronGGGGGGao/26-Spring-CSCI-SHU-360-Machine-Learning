"""
Feature ablation branch for the Stage 2 5-day recent-window MLP.

This branch keeps the existing recent-window target, train/validation/test
methodology, and portfolio construction, then adds a small set of features that
are explicitly aligned with a 5-trading-day forecast horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlp.day5_stage2_recent_window.features import (  # noqa: F401
    EXCESS_TARGET_COLUMN,
    FORWARD_HORIZON,
    RAW_RETURN_COLUMN,
    TARGET_COLUMN,
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    build_features as build_base_features,
)


EXTRA_FEATURE_COLUMNS = [
    "ret_5d_lag1",
    "ret_5d_lag2",
    "excess_ret_5d_lag1",
    "excess_ret_5d_lag2",
    "amount_z_5d",
    "turnover_z_5d",
    "range_5d",
    "close_pos_5d_range",
    "ret_5d_rank",
    "amount_z_5d_rank",
    "turnover_z_5d_rank",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS


def _add_per_stock_5d_features(df: pd.DataFrame) -> pd.DataFrame:
    group_name = getattr(df, "name", None)
    df = df.sort_values("date").copy()
    if group_name is not None:
        df["stock_code"] = group_name

    high_5d = df["high"].astype(float).rolling(5).max()
    low_5d = df["low"].astype(float).rolling(5).min()
    range_span = (high_5d - low_5d).replace(0, np.nan)
    amount = df["amount"].astype(float)
    amount_std_5d = amount.rolling(5).std().replace(0, np.nan)

    df["ret_5d_lag1"] = df["ret_5d"].shift(1)
    df["ret_5d_lag2"] = df["ret_5d"].shift(2)
    df["excess_ret_5d_lag1"] = df["excess_ret_5d"].shift(1)
    df["excess_ret_5d_lag2"] = df["excess_ret_5d"].shift(2)
    df["amount_z_5d"] = (amount - amount.rolling(5).mean()) / amount_std_5d
    df["range_5d"] = high_5d / low_5d - 1.0
    df["close_pos_5d_range"] = (df["close"].astype(float) - low_5d) / range_span

    if "turnover" in df.columns:
        turnover = df["turnover"].astype(float)
        turnover_std_5d = turnover.rolling(5).std().replace(0, np.nan)
        df["turnover_z_5d"] = (turnover - turnover.rolling(5).mean()) / turnover_std_5d
    else:
        df["turnover_z_5d"] = np.nan
    return df


def _add_cross_sectional_5d_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    rank_bases = ["ret_5d", "amount_z_5d", "turnover_z_5d"]
    for base in rank_bases:
        panel[f"{base}_rank"] = panel.groupby("date")[base].rank(method="average", pct=True)
    return panel


def build_features(prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    panel = build_base_features(prices, index_df)
    panel = panel.groupby("stock_code", group_keys=True).apply(_add_per_stock_5d_features)
    if isinstance(panel.index, pd.MultiIndex):
        if "stock_code" in panel.columns:
            panel = panel.reset_index(level=0, drop=True)
        else:
            panel = panel.reset_index(level=0).rename(columns={"level_0": "stock_code"})
    panel = panel.reset_index(drop=True)
    return _add_cross_sectional_5d_ranks(panel)


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, RAW_RETURN_COLUMN]).copy()
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
