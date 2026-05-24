"""
Recent-window search on top of the current tuned 3-day MLP.

This branch keeps the tuned MLP architecture, features, and portfolio rules,
but restricts training to the most recent N trading days to test whether older
regimes are diluting short-horizon signal.
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
    from .features import FEATURE_COLUMNS, FORWARD_HORIZON, build_features, prediction_frame, training_frame
except ImportError:  # noqa: E402
    from features import FEATURE_COLUMNS, FORWARD_HORIZON, build_features, prediction_frame, training_frame

from mlp.day3_stage1_tuned.mlp_model import (  # noqa: E402
    MLP_CONFIGS,
    evaluate_model,
    parse_int_list,
    parse_str_list,
    select_model_and_portfolio,
)

VAL_DAYS = 10
DEFAULT_LOOKBACKS = ["63", "126", "189", "252", "full"]


def parse_lookbacks(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def _restrict_recent_window(train_df: pd.DataFrame, lookback: str) -> tuple[pd.DataFrame, str]:
    if lookback == "full":
        return train_df.copy(), "full"
    lookback_days = int(lookback)
    train_dates = np.sort(train_df["date"].unique())
    if len(train_dates) < lookback_days:
        return train_df.copy(), "full_fallback"
    start_date = pd.Timestamp(train_dates[-lookback_days])
    return train_df[train_df["date"] >= start_date].copy(), start_date.date().isoformat()


def build_dev_split(panel: pd.DataFrame, as_of: str | None = None):
    as_of_ts = pd.Timestamp(as_of) if as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    return {
        "train_df": train_pool[train_pool["date"] <= train_end].copy(),
        "val_df": train_pool[train_pool["date"] >= val_start].copy(),
        "train_end": train_end,
        "val_start": val_start,
    }


def select_with_lookback(train_df, val_df, index_df, configs, top_ks, methods, lookbacks):
    leaderboard = []
    best_key = None
    best = None
    for lookback in lookbacks:
        recent_train, start_marker = _restrict_recent_window(train_df, lookback)
        unique_dates = recent_train["date"].nunique()
        if unique_dates < 40 or len(recent_train) < 5000:
            leaderboard.append(
                {
                    "lookback": lookback,
                    "train_window_start": start_marker,
                    "skipped": True,
                    "reason": "insufficient_recent_training_data",
                    "train_rows": int(len(recent_train)),
                    "train_dates": int(unique_dates),
                }
            )
            continue

        model, best_config, best_top_k, best_method, metrics, inner_rows = select_model_and_portfolio(
            recent_train, val_df, index_df, configs, top_ks, methods
        )
        for row in inner_rows:
            leaderboard.append(
                {
                    "lookback": lookback,
                    "train_window_start": start_marker,
                    "train_rows": int(len(recent_train)),
                    "train_dates": int(unique_dates),
                    **row,
                }
            )

        key = (
            metrics.get("mean_excess_return", float("-inf")),
            metrics.get("positive_excess_rate", float("-inf")),
            metrics.get("rank_ic", float("-inf")),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (
                model,
                lookback,
                start_marker,
                int(len(recent_train)),
                int(unique_dates),
                best_config,
                int(best_top_k),
                best_method,
                metrics,
            )
    if best is None:
        raise RuntimeError("All recent-window candidates were skipped; not enough training data.")
    return (*best, leaderboard)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35")
    p.add_argument(
        "--weight-methods",
        default="softmax_1.0,softmax_1.2,softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.2_0.25,softmax_risk_1.5_0.25,softmax_risk_1.5_0.50",
    )
    p.add_argument("--lookbacks", default="63,126,189,252,full")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building recent-window feature panel")
    panel = build_features(prices, index_df)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df, val_df = splits["train_df"], splits["val_df"]
    print(f"   base train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()}")

    print(">> Training recent-window tuned MLP")
    (
        model,
        best_lookback,
        best_start_marker,
        best_train_rows,
        best_train_dates,
        best_config,
        best_top_k,
        best_method,
        val_metrics,
        leaderboard,
    ) = select_with_lookback(train_df, val_df, index_df, MLP_CONFIGS, top_ks, methods, lookbacks)
    print(
        "   selected lookback/config/top_k/weight: "
        f"{best_lookback} ({best_start_marker}) / {best_config} / {best_top_k} / {best_method}"
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
        print(f"   validation positive excess rate: {val_metrics['positive_excess_rate']*100:.1f}% over {int(val_metrics['n_dates'])} dates")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of).copy()
    pred_date = pred_df["date"].iloc[0]
    pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS])
    from mlp.day3_stage1_tuned.mlp_model import build_portfolio_custom as build_portfolio  # noqa: E402
    weights = build_portfolio(pred_df, top_k=best_top_k, method=best_method)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if args.json_out:
        payload = {
            "selected_lookback": best_lookback,
            "selected_train_window_start": best_start_marker,
            "selected_train_rows": best_train_rows,
            "selected_train_dates": best_train_dates,
            "selected_config": best_config,
            "selected_top_k": best_top_k,
            "selected_weight_method": best_method,
            "model_configs": MLP_CONFIGS,
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "leaderboard": leaderboard,
            "validation": val_metrics,
            "prediction_date": pred_date.date().isoformat(),
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f">> Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
