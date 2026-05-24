"""
Canonical self-test for the Stage 2 5-day recent/style score blend.
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
from mlp.day5_stage2_style_recent_blend.mlp_model import (  # noqa: E402
    FORWARD_HORIZON,
    RECENT_CONFIGS,
    STYLE_CONFIGS,
    _branch_frames,
    _score_frame,
    _split_bounds,
    build_recent_features,
    build_style_features,
    evaluate_blend,
    parse_float_list,
    parse_half_lives,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    select_blend,
    train_branch_models,
)


VAL_DAYS = 10
TEST_DAYS = 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument("--weight-methods", default="softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.5_0.50")
    p.add_argument("--lookbacks", default="126,189,252")
    p.add_argument("--half-lives", default="10,20,40")
    p.add_argument("--style-policies", default="static_softmax_2.0,trend_dynamic,breadth_dynamic,defensive_dynamic")
    p.add_argument("--blend-weights", default="0.25,0.40,0.50,0.60,0.75")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    half_lives = parse_half_lives(args.half_lives)
    style_policies = parse_str_list(args.style_policies)
    blend_weights = parse_float_list(args.blend_weights)

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    bounds = _split_bounds(recent_panel, val_days=VAL_DAYS, test_days=TEST_DAYS, as_of=args.as_of)
    frames = _branch_frames(recent_panel, style_panel, bounds, include_test=True)

    print(">> Stage 2 5-day blend self-test split")
    print(f"   train: {len(frames['recent_train']):,} rows up to {bounds['train_end'].date()}")
    print(f"   val:   {len(frames['recent_val']):,} rows from {bounds['val_start'].date()} to {bounds['val_end'].date()}")
    print(f"   test:  {len(frames['recent_test']):,} rows from {bounds['test_start'].date()} to {bounds['test_end'].date()}")
    print(f"   horizon/embargo: {FORWARD_HORIZON} / {EMBARGO_DAYS} trading days")

    print(">> Training recent-window and style-dynamic branches")
    branch = train_branch_models(frames, index_df, top_ks, methods, lookbacks, half_lives, style_policies)
    recent_model = branch["recent"][0]
    style_model = branch["style"][0]

    print(">> Selecting validation blend")
    scored_val = _score_frame(recent_model, style_model, frames["recent_val"], frames["style_val"])
    alpha, best_top_k, best_method, val_metrics, leaderboard = select_blend(
        scored_val, index_df, blend_weights, top_ks, methods
    )

    scored_test = _score_frame(recent_model, style_model, frames["recent_test"], frames["style_test"])
    test_metrics = evaluate_blend(scored_test, index_df, alpha=alpha, top_k=best_top_k, method=best_method)

    print(f"   selected blend_recent_weight/top_k/weight: {alpha:.2f} / {best_top_k} / {best_method}")
    print(f"   recent branch: {branch['recent'][1]} / {branch['recent'][5]} / {branch['recent'][6]} / {branch['recent'][7]}")
    print(f"   style branch: {branch['style'][1]} / {branch['style'][2]} / {branch['style'][3]}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
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
            "selected_blend_recent_weight": alpha,
            "selected_top_k": best_top_k,
            "selected_weight_method": best_method,
            "branch_recent": {
                "selected_lookback": branch["recent"][1],
                "selected_train_window_start": branch["recent"][2],
                "selected_train_rows": branch["recent"][3],
                "selected_train_dates": branch["recent"][4],
                "selected_config": branch["recent"][5],
                "selected_top_k": branch["recent"][6],
                "selected_weight_method": branch["recent"][7],
                "validation": branch["recent"][8],
            },
            "branch_style": {
                "selected_config": branch["style"][1],
                "selected_half_life": branch["style"][2],
                "selected_allocation_policy": branch["style"][3],
                "validation": branch["style"][4],
            },
            "model_configs": {"recent": RECENT_CONFIGS, "style": STYLE_CONFIGS},
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "half_lives": half_lives,
            "style_policies": style_policies,
            "blend_weights": blend_weights,
            "leaderboard": leaderboard,
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
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote self-test summary to {out_path}")


if __name__ == "__main__":
    main()
