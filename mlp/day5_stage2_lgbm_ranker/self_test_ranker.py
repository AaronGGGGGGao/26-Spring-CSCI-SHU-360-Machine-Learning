"""Canonical self-test for the Stage 2 LightGBM ranker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.paths import DATA_DIR  # noqa: E402
from mlp.day5_stage2_lgbm_ranker.ranker_model import (  # noqa: E402
    FEATURE_COLUMNS,
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
    split_bounds,
    _json_safe,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--val-days", type=int, default=10)
    p.add_argument("--test-days", type=int, default=10)
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
    bounds = split_bounds(panel, val_days=args.val_days, test_days=args.test_days, as_of=args.as_of)
    frames = frames_from_bounds(panel, bounds, include_test=True)

    print(">> LightGBM ranker canonical self-test split")
    print(f"   train: {len(frames['train']):,} rows up to {bounds['train_end'].date()}")
    print(f"   val:   {len(frames['val']):,} rows from {bounds['val_start'].date()} to {bounds['val_end'].date()}")
    print(f"   test:  {len(frames['test']):,} rows from {bounds['test_start'].date()} to {bounds['test_end'].date()}")
    print(f"   features/horizon: {len(FEATURE_COLUMNS)} / {FORWARD_HORIZON}")

    print(">> Training LightGBM ranker")
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
        "   validation mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
        f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
        f"{val_metrics['mean_excess_return']*100:+.3f}%"
    )
    print(
        "   test mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
        f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
        f"{test_metrics['mean_excess_return']*100:+.3f}%"
    )
    print(
        f"   test positive/rank_ic/std/min: "
        f"{test_metrics['positive_excess_rate']*100:.1f}% / {test_metrics['rank_ic']:.4f} / "
        f"{test_metrics['excess_std']*100:.3f}% / {test_metrics['min_excess_return']*100:+.3f}%"
    )

    if args.json_out:
        payload = {
            "dataset": "data",
            "model_family": "day5_stage2_lgbm_ranker",
            "methodology": {
                "type": "canonical_self_test",
                "val_days": args.val_days,
                "test_days": args.test_days,
                "forward_horizon": int(FORWARD_HORIZON),
                "target_column": TARGET_COLUMN,
            },
            "split": {
                "train_end": bounds["train_end"].date().isoformat(),
                "val_start": bounds["val_start"].date().isoformat(),
                "val_end": bounds["val_end"].date().isoformat(),
                "test_start": bounds["test_start"].date().isoformat(),
                "test_end": bounds["test_end"].date().isoformat(),
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
            "validation": val_metrics,
            "test": test_metrics,
            "leaderboard": leaderboard,
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f">> Wrote self-test summary to {out}")


if __name__ == "__main__":
    main()
