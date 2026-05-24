"""
Robust self-test for the recent-window style-dynamic 3-day tuned MLP.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
try:  # noqa: E402
    from .features import FORWARD_HORIZON, build_features, training_frame
except ImportError:  # noqa: E402
    from features import FORWARD_HORIZON, build_features, training_frame

from mlp.day3_stage1_recent_style_dynamic.mlp_model import parse_lookbacks, select_with_lookback  # noqa: E402
from mlp.day3_stage1_style_dynamic.mlp_model import (  # noqa: E402
    ALLOCATION_POLICIES,
    MLP_CONFIGS,
    evaluate_model,
    parse_int_list,
)

VAL_DAYS = 15
TEST_DAYS = 20


def build_splits(panel: pd.DataFrame, as_of: str | None = None):
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(pool["date"].unique())
    need = TEST_DAYS + VAL_DAYS + 2 * EMBARGO_DAYS + 30
    if len(all_dates) < need:
        raise RuntimeError(
            f"Not enough dates for robust self-test split: need at least {need}, got {len(all_dates)}."
        )

    test_start = pd.Timestamp(all_dates[-TEST_DAYS])
    val_end_idx = -(TEST_DAYS + EMBARGO_DAYS + 1)
    val_end = pd.Timestamp(all_dates[val_end_idx])
    val_start = pd.Timestamp(all_dates[val_end_idx - VAL_DAYS + 1])
    train_end_idx = -(TEST_DAYS + EMBARGO_DAYS + VAL_DAYS + EMBARGO_DAYS + 1)
    train_end = pd.Timestamp(all_dates[train_end_idx])
    return {
        "train_df": pool[pool["date"] <= train_end].copy(),
        "val_df": pool[(pool["date"] >= val_start) & (pool["date"] <= val_end)].copy(),
        "test_df": pool[pool["date"] >= test_start].copy(),
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--half-lives", default="10,20,40")
    p.add_argument("--policies", default="static_softmax_2.0,trend_dynamic,breadth_dynamic,defensive_dynamic")
    p.add_argument("--lookbacks", default="63,126,189,252,full")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    half_lives = parse_int_list(args.half_lives)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    lookbacks = parse_lookbacks(args.lookbacks)

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    panel = build_features(prices, index_df)
    splits = build_splits(panel, as_of=args.as_of)
    train_df, val_df, test_df = splits["train_df"], splits["val_df"], splits["test_df"]

    print(">> Robust self-test split")
    print(f"   train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()} to {splits['val_end'].date()}")
    print(f"   test:  {len(test_df):,} rows from {splits['test_start'].date()} to {pd.Timestamp(test_df['date'].max()).date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days")

    print(">> Training recent-window style-dynamic tuned MLP")
    (
        model,
        best_lookback,
        best_start_marker,
        best_train_rows,
        best_train_dates,
        best_config,
        best_half_life,
        best_policy,
        val_metrics,
        leaderboard,
    ) = select_with_lookback(train_df, val_df, index_df, MLP_CONFIGS, half_lives, policies, lookbacks)
    test_metrics = evaluate_model(model, test_df, index_df, policy_name=best_policy)
    print(
        "   selected lookback/config/half_life/policy: "
        f"{best_lookback} ({best_start_marker}) / {best_config} / {best_half_life} / {best_policy}"
    )
    print(f"   selected recent train rows/dates: {best_train_rows:,} / {best_train_dates}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in val_metrics:
        print(
            "   validation mean 3d returns "
            f"(portfolio/benchmark/excess): "
            f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{val_metrics['mean_excess_return']*100:+.3f}%"
        )
    print(f"   test rank IC: {test_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in test_metrics:
        print(
            "   test mean 3d returns "
            f"(portfolio/benchmark/excess): "
            f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{test_metrics['mean_excess_return']*100:+.3f}%"
        )
        print(f"   test positive excess rate: {test_metrics['positive_excess_rate']*100:.1f}% over {int(test_metrics['n_dates'])} dates")

    if args.json_out:
        out = {
            "selected_lookback": best_lookback,
            "selected_train_window_start": best_start_marker,
            "selected_train_rows": best_train_rows,
            "selected_train_dates": best_train_dates,
            "selected_config": best_config,
            "selected_half_life": best_half_life,
            "selected_allocation_policy": best_policy,
            "model_configs": MLP_CONFIGS,
            "half_lives": half_lives,
            "policies": policies,
            "policy_definitions": ALLOCATION_POLICIES,
            "lookbacks": lookbacks,
            "leaderboard": leaderboard,
            "split": {
                "train_end": splits["train_end"].date().isoformat(),
                "val_start": splits["val_start"].date().isoformat(),
                "val_end": splits["val_end"].date().isoformat(),
                "test_start": splits["test_start"].date().isoformat(),
                "test_end": pd.Timestamp(test_df["date"].max()).date().isoformat(),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "forward_horizon": int(FORWARD_HORIZON),
                "embargo_days": int(EMBARGO_DAYS),
                "val_days": int(VAL_DAYS),
                "test_days": int(TEST_DAYS),
            },
            "validation": val_metrics,
            "test": test_metrics,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote robust self-test summary to {out_path}")


if __name__ == "__main__":
    main()
