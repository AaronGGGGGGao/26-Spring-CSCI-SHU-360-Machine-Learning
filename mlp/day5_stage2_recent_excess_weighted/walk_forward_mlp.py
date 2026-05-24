"""
Walk-forward robustness evaluation for the recent-window excess-target MLP.
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
try:  # noqa: E402
    from .features import FORWARD_HORIZON, TARGET_COLUMN, build_features, training_frame
    from .mlp_model import evaluate_model_policy, parse_lookbacks, parse_policies, select_with_excess_target
except ImportError:  # noqa: E402
    from features import FORWARD_HORIZON, TARGET_COLUMN, build_features, training_frame
    from mlp_model import evaluate_model_policy, parse_lookbacks, parse_policies, select_with_excess_target

from mlp.day5_stage2_tuned.mlp_model import (  # noqa: E402
    MLP_CONFIGS,
    parse_int_list,
    parse_str_list,
)


def _build_pool(panel: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    return training_frame(panel, max_date=train_cutoff)


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
    p.add_argument("--min-train-days", type=int, default=120)
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument("--weight-methods", default="softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.5_0.50")
    p.add_argument("--lookbacks", default="126,189")
    p.add_argument("--half-lives", default="20,40")
    p.add_argument("--target-transforms", default="winsor_zscore,rank")
    p.add_argument("--policies", default="static_30_softmax_2.0,static_35_softmax_2.0,static_40_softmax_risk_1.5_0.50,regime_light,regime_defensive")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    half_lives = parse_int_list(args.half_lives)
    target_transforms = parse_str_list(args.target_transforms)
    policies = parse_policies(args.policies)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    panel = build_features(prices, index_df)
    pool = _build_pool(panel, as_of=args.as_of)
    windows = build_walk_forward_windows(
        pool,
        windows=args.windows,
        val_days=args.val_days,
        test_days=args.test_days,
        embargo_days=args.embargo_days,
        min_train_days=args.min_train_days,
    )

    print(">> Walk-forward windows")
    for spec in windows:
        print(
            f"   window {spec['window_id']}: "
            f"train<= {spec['train_end'].date()} | "
            f"val {spec['val_start'].date()} to {spec['val_end'].date()} | "
            f"test {spec['test_start'].date()} to {spec['test_end'].date()}"
        )

    results: list[dict] = []
    for spec in windows:
        train_df = pool[pool["date"] <= spec["train_end"]].copy()
        val_df = pool[(pool["date"] >= spec["val_start"]) & (pool["date"] <= spec["val_end"])].copy()
        test_df = pool[(pool["date"] >= spec["test_start"]) & (pool["date"] <= spec["test_end"])].copy()

        print(f">> Training window {spec['window_id']}")
        (
            model,
            best_lookback,
            best_start_marker,
            best_train_rows,
            best_train_dates,
            best_half_life,
            best_target_transform,
            best_config,
            best_policy,
            val_metrics,
            leaderboard,
        ) = select_with_excess_target(
            train_df,
            val_df,
            index_df,
            MLP_CONFIGS,
            top_ks,
            methods,
            lookbacks,
            half_lives,
            target_transforms,
            policies,
        )
        test_metrics = evaluate_model_policy(model, test_df, index_df, policy_name=best_policy)

        print(
            "   selected lookback/half_life/target/config/policy: "
            f"{best_lookback} ({best_start_marker}) / {best_half_life} / "
            f"{best_target_transform} / {best_config} / {best_policy}"
        )
        if "mean_excess_return" in test_metrics:
            print(
                "   test mean 5d returns "
                f"(portfolio/benchmark/excess): "
                f"{test_metrics['mean_portfolio_return']*100:+.3f}% / "
                f"{test_metrics['mean_benchmark_return']*100:+.3f}% / "
                f"{test_metrics['mean_excess_return']*100:+.3f}%"
            )

        results.append(
            {
                "window_id": spec["window_id"],
                "split": {
                    "train_end": spec["train_end"].date().isoformat(),
                    "val_start": spec["val_start"].date().isoformat(),
                    "val_end": spec["val_end"].date().isoformat(),
                    "test_start": spec["test_start"].date().isoformat(),
                    "test_end": spec["test_end"].date().isoformat(),
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "test_rows": int(len(test_df)),
                },
                "selected_lookback": best_lookback,
                "selected_train_window_start": best_start_marker,
                "selected_train_rows": best_train_rows,
                "selected_train_dates": best_train_dates,
                "selected_half_life": best_half_life,
                "selected_target_transform": best_target_transform,
                "selected_config": best_config,
                "selected_allocation_policy": best_policy,
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

    print(">> Aggregate walk-forward result")
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
            "model_family": "mlp_day5_stage2_recent_excess_weighted",
            "target": TARGET_COLUMN,
            "methodology": {
                "type": "walk_forward",
                "windows": args.windows,
                "val_days": args.val_days,
                "test_days": args.test_days,
                "embargo_days": args.embargo_days,
                "min_train_days": args.min_train_days,
                "forward_horizon": int(FORWARD_HORIZON),
            },
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "half_lives": half_lives,
            "target_transforms": target_transforms,
            "policies": policies,
            "model_configs": MLP_CONFIGS,
            "window_results": results,
            "aggregate": aggregate,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote walk-forward summary to {out_path}")


if __name__ == "__main__":
    main()
