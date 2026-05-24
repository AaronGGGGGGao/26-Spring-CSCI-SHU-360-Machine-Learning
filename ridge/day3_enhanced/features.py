"""
Feature engineering for the 3-day enhanced Ridge model.

Enhancements relative to the basic version:
  - shorter-horizon momentum aligned with a 3-day target
  - intraday/range microstructure features
  - turnover and amount anomalies
  - downside volatility and volatility-term-structure features
  - market-relative features using the CSI500 index
  - more cross-sectional ranks on short-horizon signals
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "target_3d"
FORWARD_HORIZON = 3

FEATURE_COLUMNS = [
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "excess_ret_1d", "excess_ret_3d", "excess_ret_5d", "excess_ret_10d",
    "intraday_ret", "gap_1d", "range_1d", "close_pos_in_range",
    "vol_5d", "vol_10d", "vol_20d", "downside_vol_20d", "vol_term_ratio",
    "volume_z_20d", "amount_z_20d", "turnover_z_20d", "turnover_ma_10d",
    "close_over_ma10", "close_over_ma20", "close_over_ma60", "rsi_6", "rsi_14",
    "ret_3d_rank", "ret_10d_rank", "excess_ret_3d_rank", "volume_z_20d_rank",
    "turnover_z_20d_rank", "vol_10d_rank", "range_1d_rank",
]


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    group_name = getattr(df, "name", None)
    if "stock_code" not in df.columns:
        df = df.copy()
        df["stock_code"] = group_name

    df = df.sort_values("date").copy()
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    df["ret_1d"] = close.pct_change(1)
    df["ret_2d"] = close.pct_change(2)
    df["ret_3d"] = close.pct_change(3)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    df["intraday_ret"] = close / open_ - 1.0
    df["gap_1d"] = open_ / close.shift(1) - 1.0
    df["range_1d"] = high / low - 1.0
    range_span = (high - low).replace(0, np.nan)
    df["close_pos_in_range"] = (close - low) / range_span

    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    downside = df["ret_1d"].clip(upper=0)
    df["downside_vol_20d"] = downside.rolling(20).std()
    df["vol_term_ratio"] = df["vol_5d"] / df["vol_20d"]

    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    amt_mean = amount.rolling(20).mean()
    amt_std = amount.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (volume - vol_mean) / vol_std
    df["amount_z_20d"] = (amount - amt_mean) / amt_std

    if "turnover" in df.columns:
        turnover = df["turnover"].astype(float)
        to_mean = turnover.rolling(20).mean()
        to_std = turnover.rolling(20).std().replace(0, np.nan)
        df["turnover_z_20d"] = (turnover - to_mean) / to_std
        df["turnover_ma_10d"] = turnover.rolling(10).mean()
    else:
        df["turnover_z_20d"] = np.nan
        df["turnover_ma_10d"] = np.nan

    df["close_over_ma10"] = close / close.rolling(10).mean() - 1.0
    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    delta = close.diff()
    up6 = delta.clip(lower=0).rolling(6).mean()
    down6 = (-delta.clip(upper=0)).rolling(6).mean().replace(0, np.nan)
    rs6 = up6 / down6
    df["rsi_6"] = 100 - 100 / (1 + rs6)

    up14 = delta.clip(lower=0).rolling(14).mean()
    down14 = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs14 = up14 / down14
    df["rsi_14"] = 100 - 100 / (1 + rs14)

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    for base in [
        "ret_3d",
        "ret_10d",
        "excess_ret_3d",
        "volume_z_20d",
        "turnover_z_20d",
        "vol_10d",
        "range_1d",
    ]:
        panel[f"{base}_rank"] = panel.groupby("date")[base].rank(method="average", pct=True)
    return panel


def build_features(prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "stock_code", "open", "close", "high", "low", "volume", "amount"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features)
        .reset_index(drop=True)
    )

    index_panel = index_df.copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel = index_panel.sort_values("date")
    index_panel["index_ret_1d"] = index_panel["close"].pct_change(1)
    index_panel["index_ret_3d"] = index_panel["close"].pct_change(3)
    index_panel["index_ret_5d"] = index_panel["close"].pct_change(5)
    index_panel["index_ret_10d"] = index_panel["close"].pct_change(10)

    panel = panel.merge(
        index_panel[["date", "index_ret_1d", "index_ret_3d", "index_ret_5d", "index_ret_10d"]],
        on="date",
        how="left",
    )
    panel["excess_ret_1d"] = panel["ret_1d"] - panel["index_ret_1d"]
    panel["excess_ret_3d"] = panel["ret_3d"] - panel["index_ret_3d"]
    panel["excess_ret_5d"] = panel["ret_5d"] - panel["index_ret_5d"]
    panel["excess_ret_10d"] = panel["ret_10d"] - panel["index_ret_10d"]
    panel = _cross_sectional_ranks(panel)
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
