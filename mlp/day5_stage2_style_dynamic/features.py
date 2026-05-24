"""
Style- and dynamic-regime-enhanced feature panel for the Stage 2 5-day MLP.

This branch keeps the proven tuned-MLP base features and adds:
  - structured risk/style features derived from public price/turnover history
  - market-state features derived from CSI500 and cross-sectional breadth
  - a small set of interaction terms so the model can express regime-conditional
    alpha without depending on external unstable data sources
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlp.day5_stage2_tuned.features import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features as base_build_features,
)

STYLE_DYNAMIC_FEATURE_COLUMNS = [
    "beta_60d",
    "idio_vol_60d",
    "resid_mom_5d",
    "resid_mom_10d",
    "trend_eff_10d",
    "trend_eff_20d",
    "skew_20d",
    "vol_of_vol_20d",
    "liq_proxy_20d",
    "liq_shock_20d",
    "turnover_stability_20d",
    "mkt_ret_3d",
    "mkt_ret_10d",
    "mkt_ret_20d",
    "mkt_vol_5d",
    "mkt_vol_10d",
    "mkt_trend_5_20",
    "mkt_drawdown_20d",
    "mkt_breadth_5d",
    "mkt_breadth_10d",
    "mkt_dispersion_20d",
    "mkt_vol_regime",
    "beta20_x_mkt_vol",
    "residmom5_x_mkt_trend",
    "liq_x_dispersion",
    "beta_60d_rank",
    "idio_vol_60d_rank",
    "resid_mom_5d_rank",
    "trend_eff_10d_rank",
    "liq_proxy_20d_rank",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + STYLE_DYNAMIC_FEATURE_COLUMNS


def _compute_style_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["stock_code", "date"]).copy()

    def per_stock(df: pd.DataFrame) -> pd.DataFrame:
        group_name = getattr(df, "name", None)
        df = df.sort_values("date").copy()
        if "stock_code" not in df.columns:
            df["stock_code"] = group_name
        ret = df["ret_1d"].astype(float)
        idx = df["index_ret_1d"].astype(float)
        close = df["close"].astype(float)
        amount = df["amount"].astype(float)

        cov60 = ret.rolling(60).cov(idx)
        var60 = idx.rolling(60).var().replace(0, np.nan)
        df["beta_60d"] = cov60 / var60
        resid = ret - df["beta_60d"] * idx
        df["idio_vol_60d"] = resid.rolling(60).std()
        df["resid_mom_5d"] = resid.rolling(5).sum()
        df["resid_mom_10d"] = resid.rolling(10).sum()

        path10 = ret.abs().rolling(10).sum().replace(0, np.nan)
        path20 = ret.abs().rolling(20).sum().replace(0, np.nan)
        df["trend_eff_10d"] = close.pct_change(10).abs() / path10
        df["trend_eff_20d"] = close.pct_change(20).abs() / path20
        df["skew_20d"] = ret.rolling(20).skew()
        df["vol_of_vol_20d"] = df["vol_5d"].rolling(20).std()

        liq_mean_20 = amount.rolling(20).mean()
        liq_std_20 = amount.rolling(20).std().replace(0, np.nan)
        df["liq_proxy_20d"] = np.log1p(liq_mean_20.clip(lower=0))
        df["liq_shock_20d"] = (amount - liq_mean_20) / liq_std_20

        if "turnover" in df.columns:
            turnover = df["turnover"].astype(float)
            to_mean_20 = turnover.rolling(20).mean()
            to_std_20 = turnover.rolling(20).std().replace(0, np.nan)
            df["turnover_stability_20d"] = to_mean_20 / to_std_20
        else:
            df["turnover_stability_20d"] = np.nan
        return df

    panel = panel.groupby("stock_code", group_keys=False).apply(per_stock).reset_index(drop=True)
    return panel


def _compute_market_state(panel: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    close = idx["close"].astype(float)
    idx["mkt_ret_1d"] = close.pct_change(1)
    idx["mkt_ret_3d"] = close.pct_change(3)
    idx["mkt_ret_10d"] = close.pct_change(10)
    idx["mkt_ret_20d"] = close.pct_change(20)
    idx["mkt_vol_5d"] = idx["mkt_ret_1d"].rolling(5).std()
    idx["mkt_vol_10d"] = idx["mkt_ret_1d"].rolling(10).std()
    idx["mkt_trend_5_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1.0
    idx["mkt_drawdown_20d"] = close / close.rolling(20).max() - 1.0
    vol_med = idx["mkt_vol_10d"].rolling(60).median()
    idx["mkt_vol_regime"] = (idx["mkt_vol_10d"] > vol_med).astype(float)

    breadth = (
        panel.groupby("date")["ret_1d"]
        .agg(
            breadth_up=lambda s: (s > 0).mean(),
            cross_dispersion=lambda s: s.std(),
        )
        .reset_index()
        .sort_values("date")
    )
    breadth["mkt_breadth_5d"] = breadth["breadth_up"].rolling(5).mean()
    breadth["mkt_breadth_10d"] = breadth["breadth_up"].rolling(10).mean()
    breadth["mkt_dispersion_20d"] = breadth["cross_dispersion"].rolling(20).mean()

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
                "mkt_vol_regime",
            ]
        ],
        on="date",
        how="left",
    )
    panel = panel.merge(
        breadth[["date", "mkt_breadth_5d", "mkt_breadth_10d", "mkt_dispersion_20d"]],
        on="date",
        how="left",
    )
    return panel


def _interaction_and_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["beta20_x_mkt_vol"] = panel["beta_20d"] * panel["mkt_vol_10d"]
    panel["residmom5_x_mkt_trend"] = panel["resid_mom_5d"] * panel["mkt_trend_5_20"]
    panel["liq_x_dispersion"] = panel["liq_proxy_20d"] * panel["mkt_dispersion_20d"]

    rank_bases = [
        "beta_60d",
        "idio_vol_60d",
        "resid_mom_5d",
        "trend_eff_10d",
        "liq_proxy_20d",
    ]
    for base in rank_bases:
        panel[f"{base}_rank"] = panel.groupby("date")[base].rank(method="average", pct=True)
    return panel


def build_features(prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    panel = base_build_features(prices, index_df).copy()
    panel = _compute_style_features(panel)
    panel = _compute_market_state(panel, index_df)
    panel = _interaction_and_ranks(panel)
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
