"""
Self-test for the 3-day XGBoost baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import DEFAULT_TOP_K, EMBARGO_DAYS
from baseline.paths import DATA_DIR
from baseline.day3.features import FEATURE_COLUMNS, FORWARD_HORIZON, TARGET_COLUMN, build_features, training_frame
from baseline.day3.xgboost_model import build_portfolio, rank_ic, train_model, period_excess_return

VAL_DAYS = 10
TEST_DAYS = 10


def build_splits(panel: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
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

    train_df = pool[pool["date"] <= train_end].copy()
    val_df = pool[(pool["date"] >= val_start) & (pool["date"] <= val_end)].copy()
    test_df = pool[pool["date"] >= test_start].copy()
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": pd.Timestamp(test_df["date"].max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    panel = build_features(prices)
    splits = build_splits(panel, as_of=args.as_of)
    train_df = splits["train_df"]
    val_df = splits["val_df"]
    test_df = splits["test_df"]

    print(">> Self-test split")
    print(f"   train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()} to {splits['val_end'].date()}")
    print(f"   test:  {len(test_df):,} rows from {splits['test_start'].date()} to {splits['test_end'].date()}")

    print(">> Training XGBoost")
    model = train_model(train_df, val_df)
    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    test_pred = model.predict(test_df[FEATURE_COLUMNS])

    val_ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    test_ic = rank_ic(test_df[TARGET_COLUMN].to_numpy(), test_pred, test_df["date"].to_numpy())
    _, val_bt = period_excess_return(val_df, val_pred, index_df, top_k=args.top_k)
    _, test_bt = period_excess_return(test_df, test_pred, index_df, top_k=args.top_k)

    print(f"   validation rank IC: {val_ic:.4f}")
    if val_bt is not None:
        print(
            "   validation mean 3d returns "
            f"(portfolio/benchmark/excess): "
            f"{val_bt['mean_portfolio_return']*100:+.3f}% / "
            f"{val_bt['mean_benchmark_return']*100:+.3f}% / "
            f"{val_bt['mean_excess_return']*100:+.3f}%"
        )
    print(f"   test rank IC: {test_ic:.4f}")
    if test_bt is not None:
        print(
            "   test mean 3d returns "
            f"(portfolio/benchmark/excess): "
            f"{test_bt['mean_portfolio_return']*100:+.3f}% / "
            f"{test_bt['mean_benchmark_return']*100:+.3f}% / "
            f"{test_bt['mean_excess_return']*100:+.3f}%"
        )
        print(f"   test positive excess rate: {test_bt['positive_excess_rate']*100:.1f}% over {int(test_bt['n_dates'])} dates")

    if args.json_out:
        out = {
            "split": {
                "train_end": splits["train_end"].date().isoformat(),
                "val_start": splits["val_start"].date().isoformat(),
                "val_end": splits["val_end"].date().isoformat(),
                "test_start": splits["test_start"].date().isoformat(),
                "test_end": splits["test_end"].date().isoformat(),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "forward_horizon": int(FORWARD_HORIZON),
                "embargo_days": int(EMBARGO_DAYS),
            },
            "validation": {"rank_ic": float(val_ic), **(val_bt or {})},
            "test": {"rank_ic": float(test_ic), **(test_bt or {})},
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote self-test summary to {out_path}")


if __name__ == "__main__":
    main()
