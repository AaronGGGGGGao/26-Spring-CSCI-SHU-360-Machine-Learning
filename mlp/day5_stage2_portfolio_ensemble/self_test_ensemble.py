"""
Canonical self-test for the Stage 2 5-day portfolio-level ensemble.
"""
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
from mlp.day5_stage2_portfolio_ensemble.ensemble_model import (  # noqa: E402
    FORWARD_HORIZON,
    TARGET_COLUMN,
    _branch_frames,
    _feature5d_frames,
    _split_bounds,
    build_feature5d_features,
    build_feature5d_portfolios,
    build_recent_features,
    build_style_recent_portfolios,
    build_style_features,
    combine_portfolios,
    evaluate_portfolios,
    parse_float_list,
    parse_half_lives,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    select_portfolio_ensemble,
    train_components,
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
    p.add_argument("--portfolio-weights", default="0.50,0.60,0.70,0.80,0.90,1.00")
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
    portfolio_weights = parse_float_list(args.portfolio_weights)

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    feature_panel = build_feature5d_features(prices, index_df)
    bounds = _split_bounds(recent_panel, val_days=VAL_DAYS, test_days=TEST_DAYS, as_of=args.as_of)
    frames = _branch_frames(recent_panel, style_panel, bounds, include_test=True)
    feature_frames = _feature5d_frames(feature_panel, bounds, include_test=True)

    print(">> Stage 2 5-day portfolio-ensemble self-test split")
    print(f"   train: {len(frames['recent_train']):,} rows up to {bounds['train_end'].date()}")
    print(f"   val:   {len(frames['recent_val']):,} rows from {bounds['val_start'].date()} to {bounds['val_end'].date()}")
    print(f"   test:  {len(frames['recent_test']):,} rows from {bounds['test_start'].date()} to {bounds['test_end'].date()}")
    print(f"   horizon/embargo: {FORWARD_HORIZON} / {EMBARGO_DAYS} trading days")

    print(">> Training child models")
    components = train_components(
        frames,
        feature_frames,
        index_df,
        top_ks,
        methods,
        lookbacks,
        half_lives,
        style_policies,
        score_blend_weights,
    )

    style_cfg = components["style_recent"]
    feature_cfg = components["feature5d"]

    print(">> Selecting portfolio-level ensemble on validation")
    style_val_ports, _ = build_style_recent_portfolios(
        style_cfg["recent_model"],
        style_cfg["style_model"],
        frames["recent_val"],
        frames["style_val"],
        style_cfg["blend_recent_weight"],
        style_cfg["top_k"],
        style_cfg["weight_method"],
    )
    feature_val_ports, _ = build_feature5d_portfolios(
        feature_cfg["model"],
        feature_frames["feature_val"],
        feature_cfg["selected_top_k"],
        feature_cfg["selected_weight_method"],
    )
    ensemble_weight, val_metrics, leaderboard = select_portfolio_ensemble(
        style_val_ports,
        feature_val_ports,
        frames["recent_val"],
        index_df,
        portfolio_weights,
        args.vol_penalty,
        args.downside_penalty,
        args.positive_bonus,
    )

    print(">> Evaluating selected ensemble on test")
    style_test_ports, _ = build_style_recent_portfolios(
        style_cfg["recent_model"],
        style_cfg["style_model"],
        frames["recent_test"],
        frames["style_test"],
        style_cfg["blend_recent_weight"],
        style_cfg["top_k"],
        style_cfg["weight_method"],
    )
    feature_test_ports, _ = build_feature5d_portfolios(
        feature_cfg["model"],
        feature_frames["feature_test"],
        feature_cfg["selected_top_k"],
        feature_cfg["selected_weight_method"],
    )
    test_ports = combine_portfolios(style_test_ports, feature_test_ports, ensemble_weight)
    test_series, test_metrics = evaluate_portfolios(test_ports, frames["recent_test"], index_df)

    print(f"   selected style_recent_portfolio_weight: {ensemble_weight:.2f}")
    print(
        f"   style-recent branch: blend={style_cfg['blend_recent_weight']:.2f} / "
        f"top_k={style_cfg['top_k']} / {style_cfg['weight_method']}"
    )
    print(
        f"   5d-feature branch: lookback={feature_cfg['selected_lookback']} / "
        f"{feature_cfg['selected_config']} / top_k={feature_cfg['selected_top_k']} / "
        f"{feature_cfg['selected_weight_method']}"
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
        f"   test positive/std/min/names: {test_metrics['positive_excess_rate']*100:.1f}% / "
        f"{test_metrics['excess_std']*100:.3f}% / "
        f"{test_metrics['min_excess_return']*100:+.3f}% / "
        f"{test_metrics['mean_n_names']:.1f}"
    )

    if args.json_out:
        out = {
            "horizon": FORWARD_HORIZON,
            "target_column": TARGET_COLUMN,
            "selected_style_recent_portfolio_weight": ensemble_weight,
            "selection_objective": {
                "vol_penalty": args.vol_penalty,
                "downside_penalty": args.downside_penalty,
                "positive_bonus": args.positive_bonus,
            },
            "style_recent": {
                "blend_recent_weight": style_cfg["blend_recent_weight"],
                "top_k": style_cfg["top_k"],
                "weight_method": style_cfg["weight_method"],
                "validation": style_cfg["validation"],
            },
            "feature5d": {
                "selected_lookback": feature_cfg["selected_lookback"],
                "selected_train_window_start": feature_cfg["selected_train_window_start"],
                "selected_train_rows": feature_cfg["selected_train_rows"],
                "selected_train_dates": feature_cfg["selected_train_dates"],
                "selected_config": feature_cfg["selected_config"],
                "selected_top_k": feature_cfg["selected_top_k"],
                "selected_weight_method": feature_cfg["selected_weight_method"],
                "validation": feature_cfg["validation"],
            },
            "branch_recent": {
                "selected_lookback": components["branch"]["recent"][1],
                "selected_train_window_start": components["branch"]["recent"][2],
                "selected_config": components["branch"]["recent"][5],
                "selected_top_k": components["branch"]["recent"][6],
                "selected_weight_method": components["branch"]["recent"][7],
                "validation": components["branch"]["recent"][8],
            },
            "branch_style": {
                "selected_config": components["branch"]["style"][1],
                "selected_half_life": components["branch"]["style"][2],
                "selected_allocation_policy": components["branch"]["style"][3],
                "validation": components["branch"]["style"][4],
            },
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
            "validation": val_metrics,
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
            "ensemble_leaderboard": leaderboard,
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "half_lives": half_lives,
            "style_policies": style_policies,
            "score_blend_weights": score_blend_weights,
            "portfolio_weights": portfolio_weights,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote self-test summary to {out_path}")


if __name__ == "__main__":
    main()
