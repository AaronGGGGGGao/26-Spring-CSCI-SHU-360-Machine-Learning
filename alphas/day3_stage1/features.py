"""
Alpha-style factor set for stage1 3-day models.

This feature panel keeps the project on public historical price data, but moves
from a small handcrafted technical set to a denser alpha-style factor library
closer to the workflow described by the user: return transforms, up/down ratios,
streak and timing features, range position, path-shape moments, and liquidity
state features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "target_log_3d"
RAW_RETURN_COLUMN = "target_3d"
FORWARD_HORIZON = 3

FEATURE_COLUMNS = [
    "log_ret_1d", "log_ret_2d", "log_ret_3d", "log_ret_5d", "log_ret_10d", "log_ret_20d",
    "simple_ret_1d", "simple_ret_3d", "simple_ret_5d", "simple_ret_10d", "simple_ret_20d",
    "up_ratio_5", "up_ratio_10", "up_ratio_20", "up_ratio_60",
    "down_ratio_5", "down_ratio_10", "down_ratio_20",
    "days_since_up", "days_since_down", "up_streak", "down_streak",
    "ret_mean_5", "ret_mean_10", "ret_mean_20",
    "ret_std_5", "ret_std_10", "ret_std_20",
    "ret_skew_20", "ret_kurt_20",
    "gain_loss_ratio_10", "gain_loss_ratio_20",
    "intraday_ret", "gap_1d", "range_1d", "range_ma_10", "close_pos_in_range_20", "close_pos_in_range_60",
    "close_over_ma5", "close_over_ma10", "close_over_ma20", "close_over_ma60",
    "dist_high_20d", "dist_low_20d", "dist_high_60d", "dist_low_60d",
    "volume_z_10d", "volume_z_20d", "amount_z_10d", "amount_z_20d",
    "turnover_z_10d", "turnover_z_20d", "turnover_ma_5d", "turnover_ma_20d",
    "price_volume_corr_10", "price_volume_corr_20",
    "beta_20d", "idio_vol_20d", "relative_strength_3_20", "excess_ret_3d", "excess_ret_10d",
    "up_ratio_20_rank", "days_since_up_rank", "close_pos_in_range_60_rank",
    "volume_z_20d_rank", "dist_high_60d_rank", "beta_20d_rank",
]


def _rolling_days_since(mask: pd.Series) -> pd.Series:
    out = np.empty(len(mask), dtype=float)
    last_true = -1
    arr = mask.to_numpy(dtype=bool)
    for i, flag in enumerate(arr):
        if flag:
            last_true = i
            out[i] = 0.0
        else:
            out[i] = np.nan if last_true < 0 else float(i - last_true)
    return pd.Series(out, index=mask.index)


def _rolling_streak(mask: pd.Series) -> pd.Series:
    out = np.zeros(len(mask), dtype=float)
    streak = 0
    arr = mask.to_numpy(dtype=bool)
    for i, flag in enumerate(arr):
        if flag:
            streak += 1
        else:
            streak = 0
        out[i] = float(streak)
    return pd.Series(out, index=mask.index)


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    group_name = getattr(df, "name", None)
    df = df.sort_values("date").copy()
    if group_name is not None:
        df["stock_code"] = group_name

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)

    simple_ret_1d = close.pct_change(1)
    df["simple_ret_1d"] = simple_ret_1d
    df["simple_ret_3d"] = close.pct_change(3)
    df["simple_ret_5d"] = close.pct_change(5)
    df["simple_ret_10d"] = close.pct_change(10)
    df["simple_ret_20d"] = close.pct_change(20)

    log_close = np.log(close.replace(0, np.nan))
    df["log_ret_1d"] = log_close.diff(1)
    df["log_ret_2d"] = log_close.diff(2)
    df["log_ret_3d"] = log_close.diff(3)
    df["log_ret_5d"] = log_close.diff(5)
    df["log_ret_10d"] = log_close.diff(10)
    df["log_ret_20d"] = log_close.diff(20)

    up = (simple_ret_1d > 0).astype(float)
    down = (simple_ret_1d < 0).astype(float)
    df["up_ratio_5"] = up.rolling(5).mean()
    df["up_ratio_10"] = up.rolling(10).mean()
    df["up_ratio_20"] = up.rolling(20).mean()
    df["up_ratio_60"] = up.rolling(60).mean()
    df["down_ratio_5"] = down.rolling(5).mean()
    df["down_ratio_10"] = down.rolling(10).mean()
    df["down_ratio_20"] = down.rolling(20).mean()

    df["days_since_up"] = _rolling_days_since(simple_ret_1d > 0)
    df["days_since_down"] = _rolling_days_since(simple_ret_1d < 0)
    df["up_streak"] = _rolling_streak(simple_ret_1d > 0)
    df["down_streak"] = _rolling_streak(simple_ret_1d < 0)

    df["ret_mean_5"] = simple_ret_1d.rolling(5).mean()
    df["ret_mean_10"] = simple_ret_1d.rolling(10).mean()
    df["ret_mean_20"] = simple_ret_1d.rolling(20).mean()
    df["ret_std_5"] = simple_ret_1d.rolling(5).std()
    df["ret_std_10"] = simple_ret_1d.rolling(10).std()
    df["ret_std_20"] = simple_ret_1d.rolling(20).std()
    df["ret_skew_20"] = simple_ret_1d.rolling(20).skew()
    df["ret_kurt_20"] = simple_ret_1d.rolling(20).kurt()

    gains = simple_ret_1d.clip(lower=0)
    losses = (-simple_ret_1d.clip(upper=0))
    df["gain_loss_ratio_10"] = gains.rolling(10).mean() / losses.rolling(10).mean().replace(0, np.nan)
    df["gain_loss_ratio_20"] = gains.rolling(20).mean() / losses.rolling(20).mean().replace(0, np.nan)

    df["intraday_ret"] = close / open_ - 1.0
    df["gap_1d"] = open_ / close.shift(1) - 1.0
    df["range_1d"] = high / low - 1.0
    df["range_ma_10"] = df["range_1d"].rolling(10).mean()

    hi20 = high.rolling(20).max()
    lo20 = low.rolling(20).min()
    hi60 = high.rolling(60).max()
    lo60 = low.rolling(60).min()
    span20 = (hi20 - lo20).replace(0, np.nan)
    span60 = (hi60 - lo60).replace(0, np.nan)
    df["close_pos_in_range_20"] = (close - lo20) / span20
    df["close_pos_in_range_60"] = (close - lo60) / span60

    df["close_over_ma5"] = close / close.rolling(5).mean() - 1.0
    df["close_over_ma10"] = close / close.rolling(10).mean() - 1.0
    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0
    df["dist_high_20d"] = close / hi20 - 1.0
    df["dist_low_20d"] = close / lo20 - 1.0
    df["dist_high_60d"] = close / hi60 - 1.0
    df["dist_low_60d"] = close / lo60 - 1.0

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
        df["turnover_ma_20d"] = turnover.rolling(20).mean()
    else:
        df["turnover_z_10d"] = np.nan
        df["turnover_z_20d"] = np.nan
        df["turnover_ma_5d"] = np.nan
        df["turnover_ma_20d"] = np.nan

    df["price_volume_corr_10"] = close.rolling(10).corr(volume)
    df["price_volume_corr_20"] = close.rolling(20).corr(volume)

    df[RAW_RETURN_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    df[TARGET_COLUMN] = np.log(close.shift(-FORWARD_HORIZON) / close)
    return df


def _market_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    idx = df["index_ret_1d"]
    ret = df["simple_ret_1d"]
    cov20 = ret.rolling(20).cov(idx)
    var20 = idx.rolling(20).var().replace(0, np.nan)
    df["beta_20d"] = cov20 / var20
    resid = ret - df["beta_20d"] * idx
    df["idio_vol_20d"] = resid.rolling(20).std()
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    for base in [
        "up_ratio_20",
        "days_since_up",
        "close_pos_in_range_60",
        "volume_z_20d",
        "dist_high_60d",
        "beta_20d",
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
    panel = prices.groupby("stock_code", group_keys=True).apply(_per_stock_features)
    if isinstance(panel.index, pd.MultiIndex):
        if "stock_code" in panel.columns:
            panel = panel.reset_index(level=0, drop=True)
        else:
            panel = panel.reset_index(level=0).rename(columns={"level_0": "stock_code"})
    else:
        panel = panel.reset_index(drop=True)
    if "stock_code" not in panel.columns:
        raise ValueError("build_features failed to preserve stock_code")
    panel = panel.reset_index(drop=True)

    index_panel = index_df.copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel = index_panel.sort_values("date")
    index_panel["index_ret_1d"] = index_panel["close"].pct_change(1)
    index_panel["index_ret_3d"] = index_panel["close"].pct_change(3)
    index_panel["index_ret_10d"] = index_panel["close"].pct_change(10)

    panel = panel.merge(
        index_panel[["date", "index_ret_1d", "index_ret_3d", "index_ret_10d"]],
        on="date",
        how="left",
    )

    panel["excess_ret_3d"] = panel["simple_ret_3d"] - panel["index_ret_3d"]
    panel["excess_ret_10d"] = panel["simple_ret_10d"] - panel["index_ret_10d"]
    panel["relative_strength_3_20"] = panel["simple_ret_3d"] - panel["simple_ret_20d"]

    panel = panel.groupby("stock_code", group_keys=True).apply(_market_relative_features)
    if isinstance(panel.index, pd.MultiIndex):
        if "stock_code" in panel.columns:
            panel = panel.reset_index(level=0, drop=True)
        else:
            panel = panel.reset_index(level=0).rename(columns={"level_0": "stock_code"})
    panel = panel.reset_index(drop=True)
    if "stock_code" not in panel.columns:
        raise ValueError("market-relative stage dropped stock_code")
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
