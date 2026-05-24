"""
Walk-forward-optimized model-switch branch for stage1 3-day allocation.

Idea:
  - keep two already-tested alpha branches:
      1. recent-window MLP
      2. style-dynamic MLP
  - choose one branch per day using a small market-state router
  - choose the router policy by multi-window walk-forward aggregate score,
    not by one validation block alone
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402

from mlp.day5_stage2_recent_window.features import (  # noqa: E402
    FEATURE_COLUMNS as RECENT_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    build_features as build_recent_features,
    prediction_frame as recent_prediction_frame,
    training_frame as recent_training_frame,
)
from mlp.day5_stage2_recent_window.mlp_model import (  # noqa: E402
    DEFAULT_LOOKBACKS as RECENT_LOOKBACKS,
    parse_lookbacks,
    select_with_lookback,
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
    period_excess_return as style_period_excess_return,
    select_model_and_policy as select_style_model,
)
from mlp.day5_stage2_tuned.mlp_model import (  # noqa: E402
    DEFAULT_TOP_KS as RECENT_TOP_KS,
    DEFAULT_WEIGHT_METHODS as RECENT_WEIGHT_METHODS,
    MLP_CONFIGS as RECENT_CONFIGS,
    build_portfolio_custom as build_recent_portfolio,
    period_excess_return as recent_period_excess_return,
)

STYLE_HALF_LIVES = [10, 20, 40]
STYLE_POLICY_NAMES = [p["name"] for p in STYLE_POLICIES]
ROUTER_POLICIES = [
    "always_recent",
    "always_style",
    "style_in_stress",
    "style_outside_bull",
    "style_on_weak_breadth",
    "style_on_high_vol_or_drawdown",
]

ROUTER_WALKFORWARD_WINDOWS = 3
ROUTER_WALKFORWARD_VAL_DAYS = 10
ROUTER_WALKFORWARD_TEST_DAYS = 10
ROUTER_WALKFORWARD_MIN_TRAIN_DAYS = 120


def _build_dev_bounds(panel: pd.DataFrame, val_days: int = 10, as_of: str | None = None):
    as_of_ts = pd.Timestamp(as_of) if as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    pool = recent_training_frame(panel, max_date=train_cutoff)
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
    pool = recent_training_frame(panel, max_date=train_cutoff)
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


def build_walk_forward_windows(
    pool: pd.DataFrame,
    *,
    windows: int,
    val_days: int,
    test_days: int,
    embargo_days: int,
    min_train_days: int,
) -> list[dict]:
    all_dates = np.sort(pool["date"].unique())
    required = windows * test_days + windows * val_days + windows * 2 * embargo_days + min_train_days
    if len(all_dates) < required:
        raise RuntimeError(
            f"Not enough dates for walk-forward evaluation: need at least {required}, got {len(all_dates)}."
        )

    specs: list[dict] = []
    cursor = len(all_dates) - 1
    for window_idx in range(windows):
        test_end_idx = cursor
        test_start_idx = test_end_idx - test_days + 1
        val_end_idx = test_start_idx - embargo_days - 1
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - embargo_days - 1

        if train_end_idx + 1 < min_train_days:
            raise RuntimeError(
                f"Window {window_idx + 1} would leave only {train_end_idx + 1} train dates; "
                f"minimum required is {min_train_days}."
            )

        specs.append(
            {
                "window_id": window_idx + 1,
                "train_end": pd.Timestamp(all_dates[train_end_idx]),
                "val_start": pd.Timestamp(all_dates[val_start_idx]),
                "val_end": pd.Timestamp(all_dates[val_end_idx]),
                "test_start": pd.Timestamp(all_dates[test_start_idx]),
                "test_end": pd.Timestamp(all_dates[test_end_idx]),
            }
        )
        cursor = test_start_idx - 1
    return list(reversed(specs))


def _slice_branch_frames(panel: pd.DataFrame, branch: str, bounds: dict, include_test: bool = False):
    if branch == "recent":
        pool = recent_training_frame(panel, max_date=bounds["train_cutoff"])
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


def _recent_daily_returns(frame: pd.DataFrame, model, index_df: pd.DataFrame, top_k: int, method: str) -> pd.DataFrame:
    pred = model.predict(frame[RECENT_FEATURE_COLUMNS])
    result, _ = recent_period_excess_return(frame, pred, index_df, top_k=top_k, weight_method=method)
    return result


def _style_daily_returns(frame: pd.DataFrame, model, index_df: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    pred = model.predict(frame[STYLE_FEATURE_COLUMNS])
    result, _ = style_period_excess_return(frame, pred, index_df, policy_name=policy_name)
    return result


def _router_choice(router_name: str, regime_row: pd.Series) -> str:
    if router_name == "always_recent":
        return "recent"
    if router_name == "always_style":
        return "style"
    state = _style_market_state(pd.DataFrame([regime_row]))
    if router_name == "style_in_stress":
        return "style" if state == "stress" else "recent"
    if router_name == "style_outside_bull":
        return "style" if state != "bull" else "recent"
    if router_name == "style_on_weak_breadth":
        return "style" if float(regime_row["mkt_breadth_5d"]) < 0.50 or float(regime_row["mkt_ret_3d"]) < 0 else "recent"
    if router_name == "style_on_high_vol_or_drawdown":
        return "style" if float(regime_row["mkt_vol_regime"]) > 0.5 or float(regime_row["mkt_drawdown_20d"]) <= -0.04 else "recent"
    raise ValueError(f"unknown router policy: {router_name}")


def _regime_snapshot(style_frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "mkt_ret_3d",
        "mkt_ret_10d",
        "mkt_ret_20d",
        "mkt_vol_5d",
        "mkt_vol_10d",
        "mkt_breadth_5d",
        "mkt_breadth_10d",
        "mkt_trend_5_20",
        "mkt_drawdown_20d",
        "mkt_vol_regime",
    ]
    return style_frame[cols].groupby("date", as_index=False).median().sort_values("date")


def _route_returns(recent_returns: pd.DataFrame, style_returns: pd.DataFrame, regime_df: pd.DataFrame, router_name: str) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = recent_returns.merge(
        style_returns,
        on="date",
        suffixes=("_recent", "_style"),
        how="inner",
    ).merge(regime_df, on="date", how="inner")
    if merged.empty:
        raise RuntimeError("Router evaluation produced no overlapping dates between recent/style branches.")
    rows = []
    style_days = 0
    for _, row in merged.iterrows():
        branch = _router_choice(router_name, row)
        if branch == "recent":
            rows.append(
                {
                    "date": pd.Timestamp(row["date"]),
                    "branch": "recent",
                    "portfolio_return": float(row["portfolio_return_recent"]),
                    "benchmark_return": float(row["benchmark_return_recent"]),
                    "excess_return": float(row["excess_return_recent"]),
                }
            )
        else:
            style_days += 1
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
        "style_share": float(style_days / len(result)),
    }
    return result, metrics


def _fit_recent_branch(train_df: pd.DataFrame, val_df: pd.DataFrame, index_df: pd.DataFrame):
    return select_with_lookback(
        train_df,
        val_df,
        index_df,
        RECENT_CONFIGS,
        RECENT_TOP_KS,
        RECENT_WEIGHT_METHODS,
        parse_lookbacks(",".join(RECENT_LOOKBACKS)),
    )


def _fit_style_branch(train_df: pd.DataFrame, val_df: pd.DataFrame, index_df: pd.DataFrame):
    return select_style_model(
        train_df,
        val_df,
        index_df,
        STYLE_CONFIGS,
        STYLE_HALF_LIVES,
        STYLE_POLICY_NAMES,
    )


def _window_objective(window_metrics: list[dict]) -> dict[str, float]:
    mean_excesses = np.array([m["mean_excess_return"] for m in window_metrics], dtype=float)
    positive_rates = np.array([m["positive_excess_rate"] for m in window_metrics], dtype=float)
    style_shares = np.array([m["style_share"] for m in window_metrics], dtype=float)
    aggregate_mean = float(mean_excesses.mean())
    aggregate_std = float(mean_excesses.std(ddof=0))
    negative_windows = int((mean_excesses <= 0).sum())
    objective = aggregate_mean - 0.50 * aggregate_std - 0.0030 * negative_windows
    return {
        "walk_forward_mean_excess_return": aggregate_mean,
        "walk_forward_std_excess_return": aggregate_std,
        "walk_forward_positive_excess_rate": float(positive_rates.mean()),
        "walk_forward_style_share": float(style_shares.mean()),
        "negative_window_count": float(negative_windows),
        "objective": float(objective),
    }


def _choose_router_by_walk_forward(recent_panel: pd.DataFrame, style_panel: pd.DataFrame, index_df: pd.DataFrame, as_of: str | None):
    bounds = _build_dev_bounds(recent_panel, val_days=ROUTER_WALKFORWARD_VAL_DAYS, as_of=as_of)
    pool = recent_training_frame(recent_panel, max_date=bounds["train_cutoff"])
    windows = build_walk_forward_windows(
        pool,
        windows=ROUTER_WALKFORWARD_WINDOWS,
        val_days=ROUTER_WALKFORWARD_VAL_DAYS,
        test_days=ROUTER_WALKFORWARD_TEST_DAYS,
        embargo_days=EMBARGO_DAYS,
        min_train_days=ROUTER_WALKFORWARD_MIN_TRAIN_DAYS,
    )

    router_rows = []
    best_row = None
    for router_name in ROUTER_POLICIES:
        window_rows = []
        for spec in windows:
            window_bounds = {
                "train_cutoff": spec["test_end"],
                "train_end": spec["train_end"],
                "val_start": spec["val_start"],
                "val_end": spec["val_end"],
                "test_start": spec["test_start"],
                "test_end": spec["test_end"],
            }
            recent_split = _slice_branch_frames(recent_panel, "recent", window_bounds, include_test=True)
            style_split = _slice_branch_frames(style_panel, "style", window_bounds, include_test=True)

            (
                recent_model,
                recent_lookback,
                recent_start_marker,
                recent_train_rows,
                recent_train_dates,
                recent_config,
                recent_top_k,
                recent_method,
                recent_val_metrics,
                _recent_leaderboard,
            ) = _fit_recent_branch(recent_split["train_df"], recent_split["val_df"], index_df)
            (
                style_model,
                style_config,
                style_half_life,
                style_policy,
                style_val_metrics,
                _style_leaderboard,
            ) = _fit_style_branch(style_split["train_df"], style_split["val_df"], index_df)

            recent_test_returns = _recent_daily_returns(
                recent_split["test_df"], recent_model, index_df, recent_top_k, recent_method
            )
            style_test_returns = _style_daily_returns(style_split["test_df"], style_model, index_df, style_policy)
            test_regime = _regime_snapshot(style_split["test_df"])
            _, test_metrics = _route_returns(recent_test_returns, style_test_returns, test_regime, router_name)
            window_rows.append(
                {
                    "window_id": int(spec["window_id"]),
                    "split": {
                        "train_end": spec["train_end"].date().isoformat(),
                        "val_start": spec["val_start"].date().isoformat(),
                        "val_end": spec["val_end"].date().isoformat(),
                        "test_start": spec["test_start"].date().isoformat(),
                        "test_end": spec["test_end"].date().isoformat(),
                    },
                    "recent_selection": {
                        "lookback": recent_lookback,
                        "train_window_start": recent_start_marker,
                        "train_rows": int(recent_train_rows),
                        "train_dates": int(recent_train_dates),
                        "config": recent_config,
                        "top_k": int(recent_top_k),
                        "weight_method": recent_method,
                        "validation": recent_val_metrics,
                    },
                    "style_selection": {
                        "config": style_config,
                        "half_life": int(style_half_life),
                        "allocation_policy": style_policy,
                        "validation": style_val_metrics,
                    },
                    "test": test_metrics,
                }
            )

        summary = _window_objective([row["test"] for row in window_rows])
        row = {"router_policy": router_name, **summary, "window_results": window_rows}
        router_rows.append(row)
        key = (
            row["objective"],
            row["walk_forward_mean_excess_return"],
            row["walk_forward_positive_excess_rate"],
        )
        if best_row is None or key > (
            best_row["objective"],
            best_row["walk_forward_mean_excess_return"],
            best_row["walk_forward_positive_excess_rate"],
        ):
            best_row = row

    assert best_row is not None
    return {
        "router_policy": best_row["router_policy"],
        "validation": {
            "mean_excess_return": best_row["walk_forward_mean_excess_return"],
            "positive_excess_rate": best_row["walk_forward_positive_excess_rate"],
            "objective": best_row["objective"],
            "std_excess_return": best_row["walk_forward_std_excess_return"],
            "negative_window_count": best_row["negative_window_count"],
            "style_share": best_row["walk_forward_style_share"],
        },
        "leaderboard": router_rows,
        "window_results": best_row["window_results"],
        "methodology": {
            "type": "walk_forward_router_selection",
            "windows": ROUTER_WALKFORWARD_WINDOWS,
            "val_days": ROUTER_WALKFORWARD_VAL_DAYS,
            "test_days": ROUTER_WALKFORWARD_TEST_DAYS,
            "embargo_days": EMBARGO_DAYS,
            "min_train_days": ROUTER_WALKFORWARD_MIN_TRAIN_DAYS,
        },
    }


def fit_dev_models(prices: pd.DataFrame, index_df: pd.DataFrame, as_of: str | None = None):
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    bounds = _build_dev_bounds(recent_panel, val_days=10, as_of=as_of)
    recent_split = _slice_branch_frames(recent_panel, "recent", bounds, include_test=False)
    style_split = _slice_branch_frames(style_panel, "style", bounds, include_test=False)

    (
        recent_model,
        recent_lookback,
        recent_start_marker,
        recent_train_rows,
        recent_train_dates,
        recent_config,
        recent_top_k,
        recent_method,
        recent_val_metrics,
        recent_leaderboard,
    ) = _fit_recent_branch(recent_split["train_df"], recent_split["val_df"], index_df)
    (
        style_model,
        style_config,
        style_half_life,
        style_policy,
        style_val_metrics,
        style_leaderboard,
    ) = _fit_style_branch(style_split["train_df"], style_split["val_df"], index_df)

    router_selection = _choose_router_by_walk_forward(recent_panel, style_panel, index_df, as_of=as_of)

    return {
        "bounds": bounds,
        "recent_panel": recent_panel,
        "style_panel": style_panel,
        "recent_model": recent_model,
        "style_model": style_model,
        "recent_selection": {
            "lookback": recent_lookback,
            "train_window_start": recent_start_marker,
            "train_rows": int(recent_train_rows),
            "train_dates": int(recent_train_dates),
            "config": recent_config,
            "top_k": int(recent_top_k),
            "weight_method": recent_method,
            "validation": recent_val_metrics,
            "leaderboard": recent_leaderboard,
        },
        "style_selection": {
            "config": style_config,
            "half_life": int(style_half_life),
            "allocation_policy": style_policy,
            "validation": style_val_metrics,
            "leaderboard": style_leaderboard,
        },
        "router_selection": router_selection,
    }


def dev_submission_payload(dev_ctx: dict, as_of: str | None = None) -> tuple[pd.DataFrame, dict]:
    recent_pred = recent_prediction_frame(dev_ctx["recent_panel"], as_of=as_of).copy()
    style_pred = style_prediction_frame(dev_ctx["style_panel"], as_of=as_of).copy()
    pred_date = pd.Timestamp(recent_pred["date"].iloc[0])
    recent_pred["score"] = dev_ctx["recent_model"].predict(recent_pred[RECENT_FEATURE_COLUMNS])
    style_pred["score"] = dev_ctx["style_model"].predict(style_pred[STYLE_FEATURE_COLUMNS])

    router_name = dev_ctx["router_selection"]["router_policy"]
    regime_row = _regime_snapshot(style_pred).iloc[0]
    branch = _router_choice(router_name, regime_row)
    if branch == "recent":
        weights = build_recent_portfolio(
            recent_pred,
            top_k=dev_ctx["recent_selection"]["top_k"],
            method=dev_ctx["recent_selection"]["weight_method"],
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
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    bounds = _build_self_test_bounds(recent_panel, val_days=val_days, test_days=test_days, as_of=as_of)
    recent_split = _slice_branch_frames(recent_panel, "recent", bounds, include_test=True)
    style_split = _slice_branch_frames(style_panel, "style", bounds, include_test=True)

    (
        recent_model,
        recent_lookback,
        recent_start_marker,
        recent_train_rows,
        recent_train_dates,
        recent_config,
        recent_top_k,
        recent_method,
        recent_val_metrics,
        recent_leaderboard,
    ) = _fit_recent_branch(recent_split["train_df"], recent_split["val_df"], index_df)
    (
        style_model,
        style_config,
        style_half_life,
        style_policy,
        style_val_metrics,
        style_leaderboard,
    ) = _fit_style_branch(style_split["train_df"], style_split["val_df"], index_df)

    router_selection = _choose_router_by_walk_forward(
        recent_panel,
        style_panel,
        index_df,
        as_of=bounds["val_end"].date().isoformat(),
    )
    router_name = router_selection["router_policy"]

    recent_test_returns = _recent_daily_returns(recent_split["test_df"], recent_model, index_df, recent_top_k, recent_method)
    style_test_returns = _style_daily_returns(style_split["test_df"], style_model, index_df, style_policy)
    test_regime = _regime_snapshot(style_split["test_df"])
    _, test_metrics = _route_returns(recent_test_returns, style_test_returns, test_regime, router_name)

    return {
        "bounds": bounds,
        "recent_selection": {
            "lookback": recent_lookback,
            "train_window_start": recent_start_marker,
            "train_rows": int(recent_train_rows),
            "train_dates": int(recent_train_dates),
            "config": recent_config,
            "top_k": int(recent_top_k),
            "weight_method": recent_method,
            "validation": recent_val_metrics,
            "leaderboard": recent_leaderboard,
        },
        "style_selection": {
            "config": style_config,
            "half_life": int(style_half_life),
            "allocation_policy": style_policy,
            "validation": style_val_metrics,
            "leaderboard": style_leaderboard,
        },
        "router_selection": router_selection,
        "test": test_metrics,
        "split_rows": {
            "train_rows": int(len(recent_split["train_df"])),
            "val_rows": int(len(recent_split["val_df"])),
            "test_rows": int(len(recent_split["test_df"])),
        },
    }
