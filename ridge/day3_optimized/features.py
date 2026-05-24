"""
Feature engineering for the optimized 3-day Ridge model.

Changes relative to the earlier 3-day variants:
  - optimize directly for 3-day excess return vs CSI500
  - add more short-horizon and market-relative features
  - include simple beta / idiosyncratic-risk features
  - keep all features based only on public historical price data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "target_3d"
RAW_RETURN_COLUMN = "target_3d"
FORWARD_HORIZON = 3

FEATURE_COLUMNS = [
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "ret_1d_lag1", "ret_1d_lag2",
    "excess_ret_1d", "excess_ret_2d", "excess_ret_3d", "excess_ret_5d", "excess_ret_10d",
    "relative_strength_3_10", "relative_strength_1_5",
    "intraday_ret", "gap_1d", "range_1d", "range_vs_10d", "close_pos_in_range",
    "vol_3d", "vol_5d", "vol_10d", "vol_20d", "vol_ratio_3_20", "downside_vol_10d",
    "volume_z_10d", "volume_z_20d", "amount_z_10d", "amount_z_20d",
    "turnover_z_10d", "turnover_z_20d", "turnover_ma_5d", "turnover_ma_10d",
    "close_over_ma5", "close_over_ma10", "close_over_ma20", "close_over_ma60",
    "dist_high_20d", "dist_low_20d", "rsi_6", "rsi_14",
    "beta_20d", "idio_vol_20d", "idio_mom_3d",
    "excess_ret_1d_rank", "excess_ret_3d_rank", "relative_strength_3_10_rank",
    "volume_z_10d_rank", "turnover_z_10d_rank", "beta_20d_rank",
    "idio_vol_20d_rank", "close_pos_in_range_rank", "dist_high_20d_rank",
]


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    group_name = getattr(df, "name", None)
    df = df.sort_values("date").copy()
    if group_name is not None:
        df["stock_code"] = group_name
    elif "stock_code" not in df.columns:
        df["stock_code"] = np.nan

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)

    df["ret_1d"] = close.pct_change(1)
    df["ret_2d"] = close.pct_change(2)
    df["ret_3d"] = close.pct_change(3)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_1d_lag1"] = df["ret_1d"].shift(1)
    df["ret_1d_lag2"] = df["ret_1d"].shift(2)

    df["intraday_ret"] = close / open_ - 1.0
    df["gap_1d"] = open_ / close.shift(1) - 1.0
    df["range_1d"] = high / low - 1.0
    avg_range_10d = df["range_1d"].rolling(10).mean()
    df["range_vs_10d"] = df["range_1d"] / avg_range_10d - 1.0
    range_span = (high - low).replace(0, np.nan)
    df["close_pos_in_range"] = (close - low) / range_span

    df["vol_3d"] = df["ret_1d"].rolling(3).std()
    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_ratio_3_20"] = df["vol_3d"] / df["vol_20d"]
    df["downside_vol_10d"] = df["ret_1d"].clip(upper=0).rolling(10).std()

    vol_mean_10 = volume.rolling(10).mean()
    vol_std_10 = volume.rolling(10).std().replace(0, np.nan)
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().replace(0, np.nan)
    amt_mean_10 = amount.rolling(10).mean()
    amt_std_10 = amount.rolling(10).std().replace(0, np.nan)
    amt_mean_20 = amount.rolling(20).mean()
    amt_std_20 = amount.rolling(20).std().replace(0, np.nan)

    df["volume_z_10d"] = (volume - vol_mean_10) / vol_std_10
    df["volume_z_20d"] = (volume - vol_mean_20) / vol_std_20
    df["amount_z_10d"] = (amount - amt_mean_10) / amt_std_10
    df["amount_z_20d"] = (amount - amt_mean_20) / amt_std_20

    if "turnover" in df.columns:
        turnover = df["turnover"].astype(float)
        to_mean_10 = turnover.rolling(10).mean()
        to_std_10 = turnover.rolling(10).std().replace(0, np.nan)
        to_mean_20 = turnover.rolling(20).mean()
        to_std_20 = turnover.rolling(20).std().replace(0, np.nan)
        df["turnover_z_10d"] = (turnover - to_mean_10) / to_std_10
        df["turnover_z_20d"] = (turnover - to_mean_20) / to_std_20
        df["turnover_ma_5d"] = turnover.rolling(5).mean()
        df["turnover_ma_10d"] = turnover.rolling(10).mean()
    else:
        df["turnover_z_10d"] = np.nan
        df["turnover_z_20d"] = np.nan
        df["turnover_ma_5d"] = np.nan
        df["turnover_ma_10d"] = np.nan

    df["close_over_ma5"] = close / close.rolling(5).mean() - 1.0
    df["close_over_ma10"] = close / close.rolling(10).mean() - 1.0
    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0
    df["dist_high_20d"] = close / high.rolling(20).max() - 1.0
    df["dist_low_20d"] = close / low.rolling(20).min() - 1.0

    delta = close.diff()
    up6 = delta.clip(lower=0).rolling(6).mean()
    down6 = (-delta.clip(upper=0)).rolling(6).mean().replace(0, np.nan)
    rs6 = up6 / down6
    df["rsi_6"] = 100 - 100 / (1 + rs6)

    up14 = delta.clip(lower=0).rolling(14).mean()
    down14 = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs14 = up14 / down14
    df["rsi_14"] = 100 - 100 / (1 + rs14)

    df[RAW_RETURN_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _market_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    idx = df["index_ret_1d"]
    ret = df["ret_1d"]
    cov20 = ret.rolling(20).cov(idx)
    var20 = idx.rolling(20).var().replace(0, np.nan)
    df["beta_20d"] = cov20 / var20
    resid = ret - df["beta_20d"] * idx
    df["idio_vol_20d"] = resid.rolling(20).std()
    df["idio_mom_3d"] = resid.rolling(3).sum()
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    rank_bases = [
        "excess_ret_1d",
        "excess_ret_3d",
        "relative_strength_3_10",
        "volume_z_10d",
        "turnover_z_10d",
        "beta_20d",
        "idio_vol_20d",
        "close_pos_in_range",
        "dist_high_20d",
    ]
    for base in rank_bases:
        panel[f"{base}_rank"] = panel.groupby("date")[base].rank(method="average", pct=True)
    return panel


def build_features(prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "stock_code", "open", "close", "high", "low", "volume", "amount"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = prices.groupby("stock_code", group_keys=True).apply(_per_stock_features)
    if isinstance(panel.index, pd.MultiIndex):
        if "stock_code" in panel.columns:
            panel = panel.reset_index(level=0, drop=True)
        else:
            panel = panel.reset_index(level=0).rename(columns={"level_0": "stock_code"})
    panel = panel.reset_index(drop=True)

    index_panel = index_df.copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel = index_panel.sort_values("date")
    index_panel["index_ret_1d"] = index_panel["close"].pct_change(1)
    index_panel["index_ret_2d"] = index_panel["close"].pct_change(2)
    index_panel["index_ret_3d"] = index_panel["close"].pct_change(3)
    index_panel["index_ret_5d"] = index_panel["close"].pct_change(5)
    index_panel["index_ret_10d"] = index_panel["close"].pct_change(10)
    index_panel["index_target_3d"] = index_panel["close"].shift(-FORWARD_HORIZON) / index_panel["close"] - 1.0

    panel = panel.merge(
        index_panel[
            [
                "date",
                "index_ret_1d",
                "index_ret_2d",
                "index_ret_3d",
                "index_ret_5d",
                "index_ret_10d",
                "index_target_3d",
            ]
        ],
        on="date",
        how="left",
    )

    panel["excess_ret_1d"] = panel["ret_1d"] - panel["index_ret_1d"]
    panel["excess_ret_2d"] = panel["ret_2d"] - panel["index_ret_2d"]
    panel["excess_ret_3d"] = panel["ret_3d"] - panel["index_ret_3d"]
    panel["excess_ret_5d"] = panel["ret_5d"] - panel["index_ret_5d"]
    panel["excess_ret_10d"] = panel["ret_10d"] - panel["index_ret_10d"]
    panel["relative_strength_3_10"] = panel["excess_ret_3d"] - panel["excess_ret_10d"]
    panel["relative_strength_1_5"] = panel["excess_ret_1d"] - panel["excess_ret_5d"]
    panel["target_excess_3d"] = panel[RAW_RETURN_COLUMN] - panel["index_target_3d"]

    panel = panel.groupby("stock_code", group_keys=True).apply(_market_relative_features)
    if isinstance(panel.index, pd.MultiIndex):
        if "stock_code" in panel.columns:
            panel = panel.reset_index(level=0, drop=True)
        else:
            panel = panel.reset_index(level=0).rename(columns={"level_0": "stock_code"})
    panel = panel.reset_index(drop=True)
    panel = _cross_sectional_ranks(panel)
    return panel


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
