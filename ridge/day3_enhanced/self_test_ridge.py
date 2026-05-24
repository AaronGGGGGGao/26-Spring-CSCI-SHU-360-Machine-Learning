"""
Self-test for the 3-day enhanced Ridge model.
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
from ridge.day3_enhanced.features import FORWARD_HORIZON, build_features, training_frame  # noqa: E402
from ridge.day3_enhanced.ridge_model import (  # noqa: E402
    evaluate_model,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    select_model_and_portfolio,
)

VAL_DAYS = 10
TEST_DAYS = 10


def build_splits(panel: pd.DataFrame, as_of: str | None = None):
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(pool["date"].unique())
    need = TEST_DAYS + VAL_DAYS + 2 * EMBARGO_DAYS + 20
    if len(all_dates) < need:
        raise RuntimeError(f"Not enough dates for self-test split: need at least {need}, got {len(all_dates)}.")
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
    p.add_argument("--alphas", default="0.01,0.1,1,10,100")
    p.add_argument("--top-ks", default="30,40,50,60,80")
    p.add_argument("--weight-methods", default="equal,rank,softmax,score,rank_inv_vol")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    alphas = parse_float_list(args.alphas)
    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    panel = build_features(prices, index_df)
    splits = build_splits(panel, as_of=args.as_of)
    train_df, val_df, test_df = splits["train_df"], splits["val_df"], splits["test_df"]

    print(">> Self-test split")
    print(f"   train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()} to {splits['val_end'].date()}")
    print(f"   test:  {len(test_df):,} rows from {splits['test_start'].date()} to {pd.Timestamp(test_df['date'].max()).date()}")

    print(">> Training Ridge 3d enhanced")
    model, best_alpha, best_top_k, best_method, val_metrics, leaderboard = select_model_and_portfolio(
        train_df, val_df, index_df, alphas, top_ks, methods
    )
    test_metrics = evaluate_model(model, test_df, index_df, top_k=best_top_k, weight_method=best_method)
    print(f"   selected alpha/top_k/weight: {best_alpha:g} / {best_top_k} / {best_method}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in val_metrics:
        print("   validation mean 3d returns "
              f"(portfolio/benchmark/excess): "
              f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
              f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
              f"{val_metrics['mean_excess_return']*100:+.3f}%")
    print(f"   test rank IC: {test_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in test_metrics:
        print("   test mean 3d returns "
              f"(portfolio/benchmark/excess): "
              f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
              f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
              f"{test_metrics['mean_excess_return']*100:+.3f}%")
        print(f"   test positive excess rate: {test_metrics['positive_excess_rate']*100:.1f}% over {int(test_metrics['n_dates'])} dates")

    if args.json_out:
        out = {
            "selected_alpha": best_alpha,
            "selected_top_k": best_top_k,
            "selected_weight_method": best_method,
            "alphas": alphas,
            "top_ks": top_ks,
            "weight_methods": methods,
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
            },
            "validation": val_metrics,
            "test": test_metrics,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote self-test summary to {out_path}")


if __name__ == "__main__":
    main()
