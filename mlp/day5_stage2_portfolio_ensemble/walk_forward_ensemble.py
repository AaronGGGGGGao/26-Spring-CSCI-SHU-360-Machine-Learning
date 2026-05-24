"""
Walk-forward robustness evaluation for the Stage 2 portfolio-level ensemble.
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

from baseline.baseline_xgboost import EMBARGO_DAYS as DEFAULT_EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
from mlp.day5_stage2_portfolio_ensemble.ensemble_model import (  # noqa: E402
    FORWARD_HORIZON,
    TARGET_COLUMN,
    _branch_frames,
    _feature5d_frames,
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
from mlp.day5_stage2_style_recent_blend.mlp_model import recent_training_frame  # noqa: E402


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


def _pool(recent_panel: pd.DataFrame, as_of: str | None) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else recent_panel["date"].max()
    trading_dates = np.sort(recent_panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    return recent_training_frame(recent_panel, max_date=train_cutoff)


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

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    feature_panel = build_feature5d_features(prices, index_df)
    pool = _pool(recent_panel, args.as_of)
    windows = build_walk_forward_windows(
        pool,
        windows=args.windows,
        val_days=args.val_days,
        test_days=args.test_days,
        embargo_days=args.embargo_days,
        min_train_days=args.min_train_days,
    )

    print(">> Stage 2 5-day portfolio-ensemble walk-forward windows")
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
        frames = _branch_frames(recent_panel, style_panel, bounds, include_test=True)
        feature_frames = _feature5d_frames(feature_panel, bounds, include_test=True)

        print(f">> Training portfolio ensemble window {spec['window_id']}")
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
        _, test_metrics = evaluate_portfolios(test_ports, frames["recent_test"], index_df)

        print(f"   selected style_recent_portfolio_weight: {ensemble_weight:.2f}")
        print(
            "   test mean 5d returns "
            f"(portfolio/benchmark/excess): "
            f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{test_metrics['mean_excess_return']*100:+.3f}%"
        )
        print(f"   test positive excess rate: {test_metrics['positive_excess_rate']*100:.1f}%")

        results.append(
            {
                "window_id": spec["window_id"],
                "split": {
                    "train_end": spec["train_end"].date().isoformat(),
                    "val_start": spec["val_start"].date().isoformat(),
                    "val_end": spec["val_end"].date().isoformat(),
                    "test_start": spec["test_start"].date().isoformat(),
                    "test_end": spec["test_end"].date().isoformat(),
                    "train_rows": int(len(frames["recent_train"])),
                    "val_rows": int(len(frames["recent_val"])),
                    "test_rows": int(len(frames["recent_test"])),
                },
                "selected_style_recent_portfolio_weight": ensemble_weight,
                "style_recent": {
                    "blend_recent_weight": style_cfg["blend_recent_weight"],
                    "top_k": style_cfg["top_k"],
                    "weight_method": style_cfg["weight_method"],
                    "validation": style_cfg["validation"],
                },
                "feature5d": {
                    "selected_lookback": feature_cfg["selected_lookback"],
                    "selected_config": feature_cfg["selected_config"],
                    "selected_top_k": feature_cfg["selected_top_k"],
                    "selected_weight_method": feature_cfg["selected_weight_method"],
                    "validation": feature_cfg["validation"],
                },
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "ensemble_leaderboard": leaderboard,
            }
        )

    aggregate = {
        "windows": len(results),
        "weighted_test_mean_excess_return": weighted_metric_average(results, "mean_excess_return"),
        "weighted_test_mean_portfolio_return": weighted_metric_average(results, "mean_portfolio_return"),
        "weighted_test_mean_benchmark_return": weighted_metric_average(results, "mean_benchmark_return"),
        "weighted_test_positive_excess_rate": weighted_metric_average(results, "positive_excess_rate"),
        "weighted_test_excess_std": weighted_metric_average(results, "excess_std"),
        "weighted_test_mean_n_names": weighted_metric_average(results, "mean_n_names"),
    }

    print(">> Aggregate Stage 2 5-day portfolio-ensemble walk-forward result")
    print(
        "   weighted test mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{aggregate['weighted_test_mean_portfolio_return']*100:+.3f}% / "
        f"{aggregate['weighted_test_mean_benchmark_return']*100:+.3f}% / "
        f"{aggregate['weighted_test_mean_excess_return']*100:+.3f}%"
    )
    print(f"   weighted test positive excess rate: {aggregate['weighted_test_positive_excess_rate']*100:.1f}%")
    print(f"   weighted test mean names: {aggregate['weighted_test_mean_n_names']:.1f}")

    if args.json_out:
        out = {
            "dataset": "data",
            "model_family": "mlp_day5_stage2_portfolio_ensemble",
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
            "selection_objective": {
                "vol_penalty": args.vol_penalty,
                "downside_penalty": args.downside_penalty,
                "positive_bonus": args.positive_bonus,
            },
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "half_lives": half_lives,
            "style_policies": style_policies,
            "score_blend_weights": score_blend_weights,
            "portfolio_weights": portfolio_weights,
            "window_results": results,
            "aggregate": aggregate,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote walk-forward summary to {out_path}")


if __name__ == "__main__":
    main()
