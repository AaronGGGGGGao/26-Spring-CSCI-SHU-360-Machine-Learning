"""Canonical self-test for the Stage 2 static meta portfolio ensemble."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
from mlp.day5_stage2_static_meta_portfolio_ensemble.meta_model import (  # noqa: E402
    DEFAULT_META_WEIGHTS,
    DEFAULT_TRIPLE_WEIGHTS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    _branch_frames,
    _feature5d_frames,
    _json_safe,
    _split_bounds,
    build_feature5d_features,
    build_recent_features,
    build_style_features,
    bundle_summary,
    evaluate_static_meta,
    fit_static_meta,
    parse_float_list,
    parse_half_lives,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    parse_weight_quads,
    parse_weight_triples,
)


VAL_DAYS = 10
TEST_DAYS = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35")
    p.add_argument("--weight-methods", default="softmax_1.8,softmax_2.0")
    p.add_argument("--lookbacks", default="126,189")
    p.add_argument("--half-lives", default="10,40")
    p.add_argument("--style-policies", default="static_softmax_2.0,trend_dynamic,breadth_dynamic,defensive_dynamic")
    p.add_argument("--score-blend-weights", default="0.25,0.40,0.50,0.60,0.75")
    p.add_argument("--pair-weights", default="0.50,0.60,0.70,0.80,0.90,1.00")
    p.add_argument("--triple-weights", default=DEFAULT_TRIPLE_WEIGHTS)
    p.add_argument("--meta-weights", default=DEFAULT_META_WEIGHTS)
    p.add_argument("--vol-penalty", type=float, default=0.25)
    p.add_argument("--downside-penalty", type=float, default=0.25)
    p.add_argument("--positive-bonus", type=float, default=0.002)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    half_lives = parse_half_lives(args.half_lives)
    style_policies = parse_str_list(args.style_policies)
    score_blend_weights = parse_float_list(args.score_blend_weights)
    pair_weights = parse_float_list(args.pair_weights)
    triple_weights = parse_weight_triples(args.triple_weights)
    meta_weights = parse_weight_quads(args.meta_weights)

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    feature_panel = build_feature5d_features(prices, index_df)
    bounds = _split_bounds(recent_panel, val_days=VAL_DAYS, test_days=TEST_DAYS, as_of=args.as_of)
    frames = _branch_frames(recent_panel, style_panel, bounds, include_test=True)
    feature_frames = _feature5d_frames(feature_panel, bounds, include_test=True)

    print(">> Stage 2 5-day static-meta self-test split")
    print(f"   train: {len(frames['recent_train']):,} rows up to {bounds['train_end'].date()}")
    print(f"   val:   {len(frames['recent_val']):,} rows from {bounds['val_start'].date()} to {bounds['val_end'].date()}")
    print(f"   test:  {len(frames['recent_test']):,} rows from {bounds['test_start'].date()} to {bounds['test_end'].date()}")
    print(f"   horizon/embargo: {FORWARD_HORIZON} / {EMBARGO_DAYS} trading days")

    print(">> Training children and selecting static meta weights")
    bundle = fit_static_meta(
        frames,
        feature_frames,
        index_df,
        top_ks,
        methods,
        lookbacks,
        half_lives,
        style_policies,
        score_blend_weights,
        pair_weights,
        triple_weights,
        meta_weights,
        args.vol_penalty,
        args.downside_penalty,
        args.positive_bonus,
    )
    test_series, test_metrics = evaluate_static_meta(
        bundle,
        frames["recent_test"],
        frames["style_test"],
        feature_frames["feature_test"],
        index_df,
    )
    mw = bundle["selected_meta_weights"]
    print(f"   selected meta weights pair/triple/style/feature: {mw[0]:.2f} / {mw[1]:.2f} / {mw[2]:.2f} / {mw[3]:.2f}")
    print(
        "   test mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
        f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
        f"{test_metrics['mean_excess_return']*100:+.3f}%"
    )
    print(f"   test positive/std/min: {test_metrics['positive_excess_rate']*100:.1f}% / {test_metrics['excess_std']*100:.3f}% / {test_metrics['min_excess_return']*100:+.3f}%")

    if args.json_out:
        payload = {
            "horizon": FORWARD_HORIZON,
            "target_column": TARGET_COLUMN,
            "split": {
                "train_end": bounds["train_end"].date().isoformat(),
                "val_start": bounds["val_start"].date().isoformat(),
                "val_end": bounds["val_end"].date().isoformat(),
                "test_start": bounds["test_start"].date().isoformat(),
                "test_end": bounds["test_end"].date().isoformat(),
                "train_rows": int(len(frames["recent_train"])),
                "val_rows": int(len(frames["recent_val"])),
                "test_rows": int(len(frames["recent_test"])),
                "forward_horizon": int(FORWARD_HORIZON),
                "embargo_days": int(EMBARGO_DAYS),
            },
            "selection_objective": {
                "vol_penalty": args.vol_penalty,
                "downside_penalty": args.downside_penalty,
                "positive_bonus": args.positive_bonus,
            },
            **bundle_summary(bundle),
            "test": test_metrics,
            "test_series": [
                {
                    "date": row.date.date().isoformat(),
                    "portfolio_return": float(row.portfolio_return),
                    "benchmark_return": float(row.benchmark_return),
                    "excess_return": float(row.excess_return),
                    "n_names": int(row.n_names),
                    "max_weight": float(row.max_weight),
                }
                for row in test_series.itertuples(index=False)
            ],
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f">> Wrote self-test summary to {out_path}")


if __name__ == "__main__":
    main()
