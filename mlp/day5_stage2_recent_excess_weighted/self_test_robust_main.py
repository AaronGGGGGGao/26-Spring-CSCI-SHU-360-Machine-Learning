"""
Robust self-test for the recent-window excess-target MLP.
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
    from .features import FORWARD_HORIZON, TARGET_COLUMN, build_features, training_frame
    from .mlp_model import evaluate_model_policy, parse_lookbacks, parse_policies, select_with_excess_target
except ImportError:  # noqa: E402
    from features import FORWARD_HORIZON, TARGET_COLUMN, build_features, training_frame
    from mlp_model import evaluate_model_policy, parse_lookbacks, parse_policies, select_with_excess_target

from mlp.day5_stage2_tuned.mlp_model import (  # noqa: E402
    MLP_CONFIGS,
    parse_int_list,
    parse_str_list,
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
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument("--weight-methods", default="softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.5_0.50")
    p.add_argument("--lookbacks", default="126,189")
    p.add_argument("--half-lives", default="20,40")
    p.add_argument("--target-transforms", default="winsor_zscore,rank")
    p.add_argument("--policies", default="static_30_softmax_2.0,static_35_softmax_2.0,static_40_softmax_risk_1.5_0.50,regime_light,regime_defensive")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    half_lives = parse_int_list(args.half_lives)
    target_transforms = parse_str_list(args.target_transforms)
    policies = parse_policies(args.policies)

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

    print(">> Training recent-window excess-target MLP")
    (
        model,
        best_lookback,
        best_start_marker,
        best_train_rows,
        best_train_dates,
        best_half_life,
        best_target_transform,
        best_config,
        best_policy,
        val_metrics,
        leaderboard,
    ) = select_with_excess_target(
        train_df,
        val_df,
        index_df,
        MLP_CONFIGS,
        top_ks,
        methods,
        lookbacks,
        half_lives,
        target_transforms,
        policies,
    )
    test_metrics = evaluate_model_policy(model, test_df, index_df, policy_name=best_policy)
    print(
        "   selected lookback/half_life/target/config/policy: "
        f"{best_lookback} ({best_start_marker}) / {best_half_life} / "
        f"{best_target_transform} / {best_config} / {best_policy}"
    )
    print(f"   selected recent train rows/dates: {best_train_rows:,} / {best_train_dates}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in val_metrics:
        print(
            "   validation mean 5d returns "
            f"(portfolio/benchmark/excess): "
            f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{val_metrics['mean_excess_return']*100:+.3f}%"
        )
    print(f"   test rank IC: {test_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in test_metrics:
        print(
            "   test mean 5d returns "
            f"(portfolio/benchmark/excess): "
            f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{test_metrics['mean_excess_return']*100:+.3f}%"
        )
        print(f"   test positive excess rate: {test_metrics['positive_excess_rate']*100:.1f}% over {int(test_metrics['n_dates'])} dates")

    if args.json_out:
        out = {
            "target": TARGET_COLUMN,
            "selected_lookback": best_lookback,
            "selected_train_window_start": best_start_marker,
            "selected_train_rows": best_train_rows,
            "selected_train_dates": best_train_dates,
            "selected_half_life": best_half_life,
            "selected_target_transform": best_target_transform,
            "selected_config": best_config,
            "selected_allocation_policy": best_policy,
            "model_configs": MLP_CONFIGS,
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "half_lives": half_lives,
            "target_transforms": target_transforms,
            "policies": policies,
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
