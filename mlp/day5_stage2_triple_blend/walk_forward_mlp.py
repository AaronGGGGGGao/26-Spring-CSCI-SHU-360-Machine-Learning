"""Walk-forward robustness evaluation for the Stage 2 5-day triple score blend."""
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
from mlp.day5_stage2_triple_blend.mlp_model import (  # noqa: E402
    FEATURE5D_CONFIGS,
    FORWARD_HORIZON,
    RECENT_CONFIGS,
    STYLE_CONFIGS,
    _branch_frames,
    _score_frame,
    build_feature5d_features,
    build_recent_features,
    build_style_features,
    evaluate_blend,
    parse_half_lives,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    parse_weight_triples,
    recent_training_frame,
    select_blend,
    train_branch_models,
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
        raise RuntimeError(
            f"Not enough dates for walk-forward evaluation: need at least {required}, got {len(all_dates)}."
        )

    specs: list[dict] = []
    cursor = len(all_dates) - 1
    for window_idx in range(windows):
        test_end_idx = cursor
        test_start_idx = test_end_idx - test_days + 1
        val_end_idx = test_start_idx - embargo_days - 1
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - embargo_days - 1

        if train_end_idx + 1 < min_train_days:
            raise RuntimeError(
                f"Window {window_idx + 1} would leave only {train_end_idx + 1} train dates; "
                f"minimum required is {min_train_days}."
            )

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
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument("--weight-methods", default="softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.5_0.50")
    p.add_argument("--lookbacks", default="126,189,252")
    p.add_argument("--half-lives", default="10,20,40")
    p.add_argument("--style-policies", default="static_softmax_2.0,trend_dynamic,breadth_dynamic,defensive_dynamic")
    p.add_argument(
        "--blend-weights",
        default="0.60:0.40:0.00,0.50:0.35:0.15,0.45:0.35:0.20,0.40:0.40:0.20,0.35:0.40:0.25,0.30:0.40:0.30",
        help="Comma-separated recent:style:feature5d blend triples.",
    )
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    half_lives = parse_half_lives(args.half_lives)
    style_policies = parse_str_list(args.style_policies)
    blend_weights = parse_weight_triples(args.blend_weights)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    feature5d_panel = build_feature5d_features(prices, index_df)
    pool = _pool(recent_panel, args.as_of)
    windows = build_walk_forward_windows(
        pool,
        windows=args.windows,
        val_days=args.val_days,
        test_days=args.test_days,
        embargo_days=args.embargo_days,
        min_train_days=args.min_train_days,
    )

    print(">> Stage 2 5-day triple-blend walk-forward windows")
    for spec in windows:
        print(
            f"   window {spec['window_id']}: "
            f"train<= {spec['train_end'].date()} | "
            f"val {spec['val_start'].date()} to {spec['val_end'].date()} | "
            f"test {spec['test_start'].date()} to {spec['test_end'].date()}"
        )

    results: list[dict] = []
    for spec in windows:
        bounds = {
            "train_cutoff": pd.Timestamp(pool["date"].max()),
            "train_end": spec["train_end"],
            "val_start": spec["val_start"],
            "val_end": spec["val_end"],
            "test_start": spec["test_start"],
            "test_end": spec["test_end"],
        }
        frames = _branch_frames(recent_panel, style_panel, feature5d_panel, bounds, include_test=True)

        print(f">> Training triple-blend window {spec['window_id']}")
        branch = train_branch_models(frames, index_df, top_ks, methods, lookbacks, half_lives, style_policies)
        recent_model = branch["recent"][0]
        style_model = branch["style"][0]
        feature5d_model = branch["feature5d"][0]

        scored_val = _score_frame(
            recent_model,
            style_model,
            feature5d_model,
            frames["recent_val"],
            frames["style_val"],
            frames["feature5d_val"],
        )
        best_weights, best_top_k, best_method, val_metrics, leaderboard = select_blend(
            scored_val, index_df, blend_weights, top_ks, methods
        )
        scored_test = _score_frame(
            recent_model,
            style_model,
            feature5d_model,
            frames["recent_test"],
            frames["style_test"],
            frames["feature5d_test"],
        )
        test_metrics = evaluate_blend(scored_test, index_df, weights=best_weights, top_k=best_top_k, method=best_method)

        print(
            "   selected blend recent/style/feature5d, top_k, weight: "
            f"{best_weights[0]:.2f}/{best_weights[1]:.2f}/{best_weights[2]:.2f} / {best_top_k} / {best_method}"
        )
        if "mean_excess_return" in test_metrics:
            print(
                "   test mean 5d returns "
                f"(portfolio/benchmark/excess): "
                f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
                f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
                f"{test_metrics['mean_excess_return']*100:+.3f}%"
            )
            print(f"   test positive excess rate: {test_metrics['positive_excess_rate']*100:.1f}% over {int(test_metrics['n_dates'])} dates")

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
                "selected_blend_recent_weight": best_weights[0],
                "selected_blend_style_weight": best_weights[1],
                "selected_blend_feature5d_weight": best_weights[2],
                "selected_top_k": best_top_k,
                "selected_weight_method": best_method,
                "branch_recent": {
                    "selected_lookback": branch["recent"][1],
                    "selected_train_window_start": branch["recent"][2],
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
                "branch_feature5d": {
                    "selected_lookback": branch["feature5d"][1],
                    "selected_train_window_start": branch["feature5d"][2],
                    "selected_config": branch["feature5d"][5],
                    "selected_top_k": branch["feature5d"][6],
                    "selected_weight_method": branch["feature5d"][7],
                    "validation": branch["feature5d"][8],
                },
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
    }

    print(">> Aggregate Stage 2 5-day triple-blend walk-forward result")
    if aggregate["weighted_test_mean_excess_return"] is not None:
        print(
            "   weighted test mean 5d returns "
            f"(portfolio/benchmark/excess): "
            f"{aggregate['weighted_test_mean_portfolio_return']*100:+.3f}% / "
            f"{aggregate['weighted_test_mean_benchmark_return']*100:+.3f}% / "
            f"{aggregate['weighted_test_mean_excess_return']*100:+.3f}%"
        )
        print(f"   weighted test positive excess rate: {aggregate['weighted_test_positive_excess_rate']*100:.1f}%")
        print(f"   weighted test rank IC: {aggregate['weighted_test_rank_ic']:.4f}")

    if args.json_out:
        out = {
            "dataset": "data",
            "model_family": "mlp_day5_stage2_triple_blend",
            "methodology": {
                "type": "walk_forward",
                "windows": args.windows,
                "val_days": args.val_days,
                "test_days": args.test_days,
                "embargo_days": args.embargo_days,
                "min_train_days": args.min_train_days,
                "forward_horizon": int(FORWARD_HORIZON),
            },
            "model_configs": {"recent": RECENT_CONFIGS, "style": STYLE_CONFIGS, "feature5d": FEATURE5D_CONFIGS},
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "half_lives": half_lives,
            "style_policies": style_policies,
            "blend_weights": blend_weights,
            "window_results": results,
            "aggregate": aggregate,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote walk-forward summary to {out_path}")


if __name__ == "__main__":
    main()
