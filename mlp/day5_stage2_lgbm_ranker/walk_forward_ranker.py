"""Walk-forward evaluation for the Stage 2 LightGBM ranker."""
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

from baseline.baseline_xgboost import EMBARGO_DAYS as DEFAULT_EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
from mlp.day5_stage2_lgbm_ranker.ranker_model import (  # noqa: E402
    FORWARD_HORIZON,
    LGBM_CONFIGS,
    TARGET_COLUMN,
    build_features,
    evaluate_model,
    frames_from_bounds,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    select_ranker,
    training_frame,
    _json_safe,
)


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
        raise RuntimeError(f"Not enough dates for walk-forward: need {required}, got {len(all_dates)}.")
    specs = []
    cursor = len(all_dates) - 1
    for window_idx in range(windows):
        test_end_idx = cursor
        test_start_idx = test_end_idx - test_days + 1
        val_end_idx = test_start_idx - embargo_days - 1
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - embargo_days - 1
        if train_end_idx + 1 < min_train_days:
            raise RuntimeError(f"Window {window_idx + 1} leaves too little training history.")
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


def _pool(panel: pd.DataFrame, as_of: str | None) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    return training_frame(panel, max_date=train_cutoff)


def weighted_metric_average(results: list[dict], key: str) -> float | None:
    pairs = [(r["test_metrics"].get("n_dates"), r["test_metrics"].get(key)) for r in results]
    pairs = [(float(w), float(v)) for w, v in pairs if w is not None and v is not None and not np.isnan(v)]
    if not pairs:
        return None
    total = sum(w for w, _ in pairs)
    if total <= 0:
        return None
    return sum(w * v for w, v in pairs) / total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--val-days", type=int, default=10)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS)
    p.add_argument("--min-train-days", type=int, default=100)
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument("--weight-methods", default="softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.5_0.50")
    p.add_argument("--lookbacks", default="126,189")
    p.add_argument("--label-bins", default="5,10")
    p.add_argument("--vol-penalty", type=float, default=0.15)
    p.add_argument("--downside-penalty", type=float, default=0.20)
    p.add_argument("--positive-bonus", type=float, default=0.002)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    label_bins = parse_int_list(args.label_bins)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    panel = build_features(prices, index_df)
    pool = _pool(panel, args.as_of)
    windows = build_walk_forward_windows(
        pool,
        windows=args.windows,
        val_days=args.val_days,
        test_days=args.test_days,
        embargo_days=args.embargo_days,
        min_train_days=args.min_train_days,
    )

    print(">> Stage 2 LightGBM ranker walk-forward windows")
    for spec in windows:
        print(
            f"   window {spec['window_id']}: "
            f"train<= {spec['train_end'].date()} | "
            f"val {spec['val_start'].date()} to {spec['val_end'].date()} | "
            f"test {spec['test_start'].date()} to {spec['test_end'].date()}"
        )

    results = []
    for spec in windows:
        bounds = {
            "train_cutoff": pd.Timestamp(pool["date"].max()),
            "train_end": spec["train_end"],
            "val_start": spec["val_start"],
            "val_end": spec["val_end"],
            "test_start": spec["test_start"],
            "test_end": spec["test_end"],
        }
        frames = frames_from_bounds(panel, bounds, include_test=True)
        print(f">> Training LightGBM ranker window {spec['window_id']}")
        (
            model,
            best_lookback,
            best_start_marker,
            best_train_rows,
            best_train_dates,
            best_config,
            best_label_bins,
            best_top_k,
            best_method,
            val_metrics,
            leaderboard,
        ) = select_ranker(
            frames["train"],
            frames["val"],
            index_df,
            LGBM_CONFIGS,
            top_ks,
            methods,
            lookbacks,
            label_bins,
            args.vol_penalty,
            args.downside_penalty,
            args.positive_bonus,
        )
        test_metrics = evaluate_model(model, frames["test"], index_df, best_top_k, best_method)
        print(
            "   selected lookback/config/label_bins/top_k/weight: "
            f"{best_lookback} / {best_config['name']} / {best_label_bins} / {best_top_k} / {best_method}"
        )
        print(
            "   test mean 5d returns "
            f"(portfolio/benchmark/excess): "
            f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{test_metrics['mean_excess_return']*100:+.3f}%"
        )
        print(f"   test positive/rank_ic: {test_metrics['positive_excess_rate']*100:.1f}% / {test_metrics['rank_ic']:.4f}")

        results.append(
            {
                "window_id": spec["window_id"],
                "split": {
                    "train_end": spec["train_end"].date().isoformat(),
                    "val_start": spec["val_start"].date().isoformat(),
                    "val_end": spec["val_end"].date().isoformat(),
                    "test_start": spec["test_start"].date().isoformat(),
                    "test_end": spec["test_end"].date().isoformat(),
                    "train_rows": int(len(frames["train"])),
                    "val_rows": int(len(frames["val"])),
                    "test_rows": int(len(frames["test"])),
                },
                "selected_lookback": best_lookback,
                "selected_train_window_start": best_start_marker,
                "selected_train_rows": best_train_rows,
                "selected_train_dates": best_train_dates,
                "selected_config": best_config,
                "selected_label_bins": best_label_bins,
                "selected_top_k": best_top_k,
                "selected_weight_method": best_method,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "leaderboard": leaderboard,
            }
        )

    aggregate = {
        "windows": len(results),
        "weighted_test_mean_excess_return": weighted_metric_average(results, "mean_excess_return"),
        "weighted_test_mean_portfolio_return": weighted_metric_average(results, "mean_portfolio_return"),
        "weighted_test_mean_benchmark_return": weighted_metric_average(results, "mean_benchmark_return"),
        "weighted_test_positive_excess_rate": weighted_metric_average(results, "positive_excess_rate"),
        "weighted_test_rank_ic": weighted_metric_average(results, "rank_ic"),
        "weighted_test_excess_std": weighted_metric_average(results, "excess_std"),
    }

    print(">> Aggregate Stage 2 LightGBM ranker walk-forward result")
    print(
        "   weighted test mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{aggregate['weighted_test_mean_portfolio_return']*100:+.3f}% / "
        f"{aggregate['weighted_test_mean_benchmark_return']*100:+.3f}% / "
        f"{aggregate['weighted_test_mean_excess_return']*100:+.3f}%"
    )
    print(
        f"   weighted test positive/rank_ic: "
        f"{aggregate['weighted_test_positive_excess_rate']*100:.1f}% / "
        f"{aggregate['weighted_test_rank_ic']:.4f}"
    )

    if args.json_out:
        payload = {
            "dataset": "data",
            "model_family": "day5_stage2_lgbm_ranker",
            "methodology": {
                "type": "walk_forward",
                "windows": args.windows,
                "val_days": args.val_days,
                "test_days": args.test_days,
                "embargo_days": args.embargo_days,
                "min_train_days": args.min_train_days,
                "forward_horizon": int(FORWARD_HORIZON),
                "target_column": TARGET_COLUMN,
            },
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "label_bins": label_bins,
            "window_results": results,
            "aggregate": aggregate,
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f">> Wrote walk-forward summary to {out}")


if __name__ == "__main__":
    main()
