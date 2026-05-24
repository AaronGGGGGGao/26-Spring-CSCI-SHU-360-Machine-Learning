"""
Model-switch branch for stage1 3-day allocation.

Idea:
  - keep two already-tested alpha branches:
      1. tuned MLP
      2. style-dynamic MLP
  - choose one branch per day using a small market-state router
  - avoid simple linear blending, which already failed earlier
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402

from mlp.day5_stage2_tuned.features import (  # noqa: E402
    FEATURE_COLUMNS as TUNED_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    build_features as build_tuned_features,
    prediction_frame as tuned_prediction_frame,
    training_frame as tuned_training_frame,
)
from mlp.day5_stage2_tuned.mlp_model import (  # noqa: E402
    MLP_CONFIGS as TUNED_CONFIGS,
    build_portfolio_custom as build_tuned_portfolio,
    make_model as make_tuned_model,
    period_excess_return as tuned_period_excess_return,
    select_model_and_portfolio as select_tuned_model,
)

from mlp.day5_stage2_style_dynamic.features import (  # noqa: E402
    FEATURE_COLUMNS as STYLE_FEATURE_COLUMNS,
    build_features as build_style_features,
    prediction_frame as style_prediction_frame,
    training_frame as style_training_frame,
)
from mlp.day5_stage2_style_dynamic.mlp_model import (  # noqa: E402
    ALLOCATION_POLICIES as STYLE_POLICIES,
    MLP_CONFIGS as STYLE_CONFIGS,
    build_portfolio_custom as build_style_portfolio,
    evaluate_model as evaluate_style_model,
    period_excess_return as style_period_excess_return,
    select_model_and_policy as select_style_model,
)

TUNED_TOP_KS = [30, 35]
TUNED_WEIGHT_METHODS = [
    "softmax_1.0",
    "softmax_1.2",
    "softmax_1.5",
    "softmax_1.8",
    "softmax_2.0",
    "softmax_risk_1.2_0.25",
    "softmax_risk_1.5_0.25",
    "softmax_risk_1.5_0.50",
]
STYLE_HALF_LIVES = [10, 20, 40]
STYLE_POLICY_NAMES = [p["name"] for p in STYLE_POLICIES]
ROUTER_POLICIES = [
    "always_tuned",
    "always_style",
    "style_in_stress",
    "style_outside_bull",
    "style_on_weak_breadth",
    "style_on_high_vol_or_drawdown",
]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _build_dev_bounds(panel: pd.DataFrame, val_days: int = 10, as_of: str | None = None):
    as_of_ts = pd.Timestamp(as_of) if as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    pool = tuned_training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(pool["date"].unique())
    if len(all_dates) < val_days + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-val_days])
    val_end = pd.Timestamp(all_dates[-1])
    train_end = pd.Timestamp(all_dates[-(val_days + EMBARGO_DAYS + 1)])
    return {
        "train_cutoff": train_cutoff,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
    }


def _build_self_test_bounds(panel: pd.DataFrame, val_days: int, test_days: int, as_of: str | None = None):
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    pool = tuned_training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(pool["date"].unique())
    need = test_days + val_days + 2 * EMBARGO_DAYS + 20
    if len(all_dates) < need:
        raise RuntimeError(f"Not enough dates for self-test split: need at least {need}, got {len(all_dates)}.")
    test_start = pd.Timestamp(all_dates[-test_days])
    test_end = pd.Timestamp(all_dates[-1])
    val_end_idx = -(test_days + EMBARGO_DAYS + 1)
    val_end = pd.Timestamp(all_dates[val_end_idx])
    val_start = pd.Timestamp(all_dates[val_end_idx - val_days + 1])
    train_end_idx = -(test_days + EMBARGO_DAYS + val_days + EMBARGO_DAYS + 1)
    train_end = pd.Timestamp(all_dates[train_end_idx])
    return {
        "train_cutoff": train_cutoff,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def _slice_branch_frames(panel: pd.DataFrame, branch: str, bounds: dict, include_test: bool = False):
    if branch == "tuned":
        pool = tuned_training_frame(panel, max_date=bounds["train_cutoff"])
    else:
        pool = style_training_frame(panel, max_date=bounds["train_cutoff"])

    train_df = pool[pool["date"] <= bounds["train_end"]].copy()
    val_df = pool[(pool["date"] >= bounds["val_start"]) & (pool["date"] <= bounds["val_end"])].copy()
    out = {"train_df": train_df, "val_df": val_df}
    if include_test:
        test_df = pool[(pool["date"] >= bounds["test_start"]) & (pool["date"] <= bounds["test_end"])].copy()
        out["test_df"] = test_df
    return out


def _style_market_state(daily: pd.DataFrame) -> str:
    trend = float(daily["mkt_trend_5_20"].median())
    breadth = float(daily["mkt_breadth_10d"].median())
    drawdown = float(daily["mkt_drawdown_20d"].median())
    vol_regime = float(daily["mkt_vol_regime"].median())
    if drawdown <= -0.07 or (vol_regime > 0.5 and breadth < 0.50) or trend < -0.01:
        return "stress"
    if trend >= 0.01 and breadth >= 0.52 and drawdown > -0.05:
        return "bull"
    return "neutral"


def _style_policy_choice(policy_name: str, daily: pd.DataFrame) -> tuple[int, str]:
    policy = next(p for p in STYLE_POLICIES if p["name"] == policy_name)
    state = _style_market_state(daily)
    choice = policy[state]
    return int(choice["top_k"]), str(choice["method"])


def _style_prediction_weights(pred_df: pd.DataFrame, policy_name: str) -> pd.Series:
    top_k, method = _style_policy_choice(policy_name, pred_df)
    return build_style_portfolio(pred_df, top_k=top_k, method=method)


def _tuned_daily_returns(frame: pd.DataFrame, model, index_df: pd.DataFrame, top_k: int, method: str) -> pd.DataFrame:
    pred = model.predict(frame[TUNED_FEATURE_COLUMNS])
    result, _ = tuned_period_excess_return(frame, pred, index_df, top_k=top_k, weight_method=method)
    return result


def _style_daily_returns(frame: pd.DataFrame, model, index_df: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    pred = model.predict(frame[STYLE_FEATURE_COLUMNS])
    result, _ = style_period_excess_return(frame, pred, index_df, policy_name=policy_name)
    return result


def _router_choice(router_name: str, regime_row: pd.Series) -> str:
    if router_name == "always_tuned":
        return "tuned"
    if router_name == "always_style":
        return "style"
    state = _style_market_state(pd.DataFrame([regime_row]))
    if router_name == "style_in_stress":
        return "style" if state == "stress" else "tuned"
    if router_name == "style_outside_bull":
        return "style" if state != "bull" else "tuned"
    if router_name == "style_on_weak_breadth":
        return "style" if float(regime_row["mkt_breadth_5d"]) < 0.50 or float(regime_row["mkt_ret_3d"]) < 0 else "tuned"
    if router_name == "style_on_high_vol_or_drawdown":
        return "style" if float(regime_row["mkt_vol_regime"]) > 0.5 or float(regime_row["mkt_drawdown_20d"]) <= -0.04 else "tuned"
    raise ValueError(f"unknown router policy: {router_name}")


def _regime_snapshot(style_frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "mkt_ret_3d",
        "mkt_breadth_5d",
        "mkt_breadth_10d",
        "mkt_trend_5_20",
        "mkt_drawdown_20d",
        "mkt_vol_regime",
    ]
    return style_frame[cols].groupby("date", as_index=False).median().sort_values("date")


def _route_returns(tuned_returns: pd.DataFrame, style_returns: pd.DataFrame, regime_df: pd.DataFrame, router_name: str) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = tuned_returns.merge(
        style_returns,
        on="date",
        suffixes=("_tuned", "_style"),
        how="inner",
    ).merge(regime_df, on="date", how="inner")
    if merged.empty:
        raise RuntimeError("Router evaluation produced no overlapping dates between tuned/style branches.")
    rows = []
    for _, row in merged.iterrows():
        branch = _router_choice(router_name, row)
        if branch == "tuned":
            rows.append(
                {
                    "date": pd.Timestamp(row["date"]),
                    "branch": "tuned",
                    "portfolio_return": float(row["portfolio_return_tuned"]),
                    "benchmark_return": float(row["benchmark_return_tuned"]),
                    "excess_return": float(row["excess_return_tuned"]),
                }
            )
        else:
            rows.append(
                {
                    "date": pd.Timestamp(row["date"]),
                    "branch": "style",
                    "portfolio_return": float(row["portfolio_return_style"]),
                    "benchmark_return": float(row["benchmark_return_style"]),
                    "excess_return": float(row["excess_return_style"]),
                }
            )
    result = pd.DataFrame(rows).sort_values("date")
    metrics = {
        "n_dates": float(len(result)),
        "mean_portfolio_return": float(result["portfolio_return"].mean()),
        "mean_benchmark_return": float(result["benchmark_return"].mean()),
        "mean_excess_return": float(result["excess_return"].mean()),
        "positive_excess_rate": float((result["excess_return"] > 0).mean()),
    }
    return result, metrics


def fit_dev_models(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: str | None = None):
    tuned_panel = build_tuned_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    bounds = _build_dev_bounds(tuned_panel, val_days=10, as_of=as_of)
    tuned_split = _slice_branch_frames(tuned_panel, "tuned", bounds, include_test=False)
    style_split = _slice_branch_frames(style_panel, "style", bounds, include_test=False)

    tuned_model, tuned_config, tuned_top_k, tuned_method, tuned_val_metrics, tuned_leaderboard = select_tuned_model(
        tuned_split["train_df"],
        tuned_split["val_df"],
        index_df,
        TUNED_CONFIGS,
        TUNED_TOP_KS,
        TUNED_WEIGHT_METHODS,
    )
    style_model, style_config, style_half_life, style_policy, style_val_metrics, style_leaderboard = select_style_model(
        style_split["train_df"],
        style_split["val_df"],
        index_df,
        STYLE_CONFIGS,
        [30, 35],
        ["softmax_1.2", "softmax_1.5", "softmax_1.8", "softmax_2.0", "softmax_risk_1.5_0.25", "score_sq"],
        STYLE_HALF_LIVES,
        [0.0, 0.5, 1.0],
    )

    tuned_returns = _tuned_daily_returns(tuned_split["val_df"], tuned_model, index_df, tuned_top_k, tuned_method)
    style_returns = _style_daily_returns(style_split["val_df"], style_model, index_df, style_policy)
    regime_df = _regime_snapshot(style_split["val_df"])

    router_rows = []
    best_router = None
    best_key = None
    for router_name in ROUTER_POLICIES:
        _, metrics = _route_returns(tuned_returns, style_returns, regime_df, router_name)
        row = {"router_policy": router_name, **metrics}
        router_rows.append(row)
        key = (
            metrics["mean_excess_return"],
            metrics["positive_excess_rate"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_router = (router_name, metrics)
    assert best_router is not None
    return {
        "bounds": bounds,
        "tuned_panel": tuned_panel,
        "style_panel": style_panel,
        "tuned_model": tuned_model,
        "style_model": style_model,
        "tuned_selection": {
            "config": tuned_config,
            "top_k": tuned_top_k,
            "weight_method": tuned_method,
            "validation": tuned_val_metrics,
            "leaderboard": tuned_leaderboard,
        },
        "style_selection": {
            "config": style_config,
            "half_life": style_half_life,
            "allocation_policy": style_policy,
            "validation": style_val_metrics,
            "leaderboard": style_leaderboard,
        },
        "router_selection": {
            "router_policy": best_router[0],
            "validation": best_router[1],
            "leaderboard": router_rows,
        },
    }


def dev_submission_payload(dev_ctx: dict, as_of: str | None = None) -> tuple[pd.DataFrame, dict]:
    tuned_pred = tuned_prediction_frame(dev_ctx["tuned_panel"], as_of=as_of).copy()
    style_pred = style_prediction_frame(dev_ctx["style_panel"], as_of=as_of).copy()
    pred_date = pd.Timestamp(tuned_pred["date"].iloc[0])
    tuned_pred["score"] = dev_ctx["tuned_model"].predict(tuned_pred[TUNED_FEATURE_COLUMNS])
    style_pred["score"] = dev_ctx["style_model"].predict(style_pred[STYLE_FEATURE_COLUMNS])

    router_name = dev_ctx["router_selection"]["router_policy"]
    regime_row = _regime_snapshot(style_pred).iloc[0]
    branch = _router_choice(router_name, regime_row)
    if branch == "tuned":
        weights = build_tuned_portfolio(
            tuned_pred,
            top_k=dev_ctx["tuned_selection"]["top_k"],
            method=dev_ctx["tuned_selection"]["weight_method"],
        )
    else:
        weights = _style_prediction_weights(style_pred, dev_ctx["style_selection"]["allocation_policy"])

    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    meta = {
        "prediction_date": pred_date.date().isoformat(),
        "selected_branch": branch,
        "selected_router_policy": router_name,
    }
    return out, meta


def fit_self_test_models(prices: pd.DataFrame, index_df: pd.DataFrame, val_days: int, test_days: int, as_of: str | None = None):
    tuned_panel = build_tuned_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    bounds = _build_self_test_bounds(tuned_panel, val_days=val_days, test_days=test_days, as_of=as_of)
    tuned_split = _slice_branch_frames(tuned_panel, "tuned", bounds, include_test=True)
    style_split = _slice_branch_frames(style_panel, "style", bounds, include_test=True)

    tuned_model, tuned_config, tuned_top_k, tuned_method, tuned_val_metrics, tuned_leaderboard = select_tuned_model(
        tuned_split["train_df"],
        tuned_split["val_df"],
        index_df,
        TUNED_CONFIGS,
        TUNED_TOP_KS,
        TUNED_WEIGHT_METHODS,
    )
    style_model, style_config, style_half_life, style_policy, style_val_metrics, style_leaderboard = select_style_model(
        style_split["train_df"],
        style_split["val_df"],
        index_df,
        STYLE_CONFIGS,
        [30, 35],
        ["softmax_1.2", "softmax_1.5", "softmax_1.8", "softmax_2.0", "softmax_risk_1.5_0.25", "score_sq"],
        STYLE_HALF_LIVES,
        [0.0, 0.5, 1.0],
    )

    tuned_val_returns = _tuned_daily_returns(tuned_split["val_df"], tuned_model, index_df, tuned_top_k, tuned_method)
    style_val_returns = _style_daily_returns(style_split["val_df"], style_model, index_df, style_policy)
    val_regime = _regime_snapshot(style_split["val_df"])

    router_rows = []
    best_router = None
    best_key = None
    for router_name in ROUTER_POLICIES:
        _, metrics = _route_returns(tuned_val_returns, style_val_returns, val_regime, router_name)
        row = {"router_policy": router_name, **metrics}
        router_rows.append(row)
        key = (
            metrics["mean_excess_return"],
            metrics["positive_excess_rate"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_router = (router_name, metrics)
    assert best_router is not None
    router_name = best_router[0]

    tuned_test_returns = _tuned_daily_returns(tuned_split["test_df"], tuned_model, index_df, tuned_top_k, tuned_method)
    style_test_returns = _style_daily_returns(style_split["test_df"], style_model, index_df, style_policy)
    test_regime = _regime_snapshot(style_split["test_df"])
    _, test_metrics = _route_returns(tuned_test_returns, style_test_returns, test_regime, router_name)

    return {
        "bounds": bounds,
        "tuned_selection": {
            "config": tuned_config,
            "top_k": tuned_top_k,
            "weight_method": tuned_method,
            "validation": tuned_val_metrics,
            "leaderboard": tuned_leaderboard,
        },
        "style_selection": {
            "config": style_config,
            "half_life": style_half_life,
            "allocation_policy": style_policy,
            "validation": style_val_metrics,
            "leaderboard": style_leaderboard,
        },
        "router_selection": {
            "router_policy": router_name,
            "validation": best_router[1],
            "leaderboard": router_rows,
        },
        "test": test_metrics,
        "split_rows": {
            "train_rows": int(len(tuned_split["train_df"])),
            "val_rows": int(len(tuned_split["val_df"])),
            "test_rows": int(len(tuned_split["test_df"])),
        },
    }
