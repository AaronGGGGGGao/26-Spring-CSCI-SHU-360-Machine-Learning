"""
Recent-window rank-ensemble stage1 MLP.

This branch keeps the current tuned MLP feature set, but trains one model per
recent lookback window and averages their cross-sectional ranks. Portfolio
selection is stability-aware: validation excess is measured across subwindows
and penalized when it is unstable.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
try:  # noqa: E402
    from .features import FEATURE_COLUMNS, FORWARD_HORIZON, TARGET_COLUMN, build_features, prediction_frame, training_frame
except ImportError:  # noqa: E402
    from features import FEATURE_COLUMNS, FORWARD_HORIZON, TARGET_COLUMN, build_features, prediction_frame, training_frame

from mlp.day5_stage2_tuned.mlp_model import (  # noqa: E402
    MLP_CONFIGS,
    build_portfolio_custom,
    make_model,
    parse_int_list,
    parse_str_list,
    period_excess_return,
    rank_ic,
)

VAL_DAYS = 10
DEFAULT_LOOKBACKS = ["126", "189", "252"]


def parse_lookbacks(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def _restrict_recent_window(train_df: pd.DataFrame, lookback: str) -> tuple[pd.DataFrame, str]:
    lookback_days = int(lookback)
    train_dates = np.sort(train_df["date"].unique())
    if len(train_dates) < lookback_days:
        return pd.DataFrame(columns=train_df.columns), "insufficient"
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


def _fit_recent_models(train_df: pd.DataFrame, config: dict, lookbacks: list[str]):
    models = []
    metadata = []
    for lookback in lookbacks:
        recent_train, start_marker = _restrict_recent_window(train_df, lookback)
        unique_dates = int(recent_train["date"].nunique()) if not recent_train.empty else 0
        if unique_dates < 40 or len(recent_train) < 5000:
            metadata.append(
                {
                    "lookback": lookback,
                    "train_window_start": start_marker,
                    "skipped": True,
                    "reason": "insufficient_recent_training_data",
                    "train_rows": int(len(recent_train)),
                    "train_dates": unique_dates,
                }
            )
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = make_model(config)
            model.fit(recent_train[FEATURE_COLUMNS], recent_train[TARGET_COLUMN])
        models.append(model)
        metadata.append(
            {
                "lookback": lookback,
                "train_window_start": start_marker,
                "train_rows": int(len(recent_train)),
                "train_dates": unique_dates,
            }
        )
    if len(models) < 2:
        return None, metadata
    return models, metadata


def predict_rank_ensemble(models: list, frame: pd.DataFrame) -> np.ndarray:
    ranked_scores = []
    dates = frame["date"]
    for model in models:
        raw = pd.Series(model.predict(frame[FEATURE_COLUMNS]), index=frame.index)
        ranked = raw.groupby(dates).rank(method="average", pct=True)
        ranked_scores.append(ranked.to_numpy(dtype=float))
    return np.mean(np.vstack(ranked_scores), axis=0)


def evaluate_ensemble(models: list, frame: pd.DataFrame, index_df: pd.DataFrame, top_k: int, weight_method: str):
    pred = predict_rank_ensemble(models, frame)
    ic = rank_ic(frame[TARGET_COLUMN].to_numpy(), pred, frame["date"].to_numpy())
    _, bt = period_excess_return(frame, pred, index_df, top_k=top_k, weight_method=weight_method)
    result = {"rank_ic": float(ic)}
    if bt is not None:
        result.update(bt)
    return result


def _validation_subwindow_metrics(
    models: list,
    val_df: pd.DataFrame,
    index_df: pd.DataFrame,
    top_k: int,
    weight_method: str,
    subwindow_days: int,
) -> dict:
    dates = np.sort(val_df["date"].unique())
    chunks = [dates[i : i + subwindow_days] for i in range(0, len(dates), subwindow_days)]
    excess_values = []
    positive_values = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        chunk_df = val_df[val_df["date"].isin(chunk)].copy()
        metrics = evaluate_ensemble(models, chunk_df, index_df, top_k=top_k, weight_method=weight_method)
        if "mean_excess_return" in metrics:
            excess_values.append(float(metrics["mean_excess_return"]))
            positive_values.append(float(metrics["positive_excess_rate"]))

    if not excess_values:
        return {
            "subwindow_count": 0,
            "subwindow_mean_excess": None,
            "subwindow_std_excess": None,
            "subwindow_positive_rate": None,
        }
    return {
        "subwindow_count": int(len(excess_values)),
        "subwindow_mean_excess": float(np.mean(excess_values)),
        "subwindow_std_excess": float(np.std(excess_values, ddof=0)),
        "subwindow_positive_rate": float(np.mean([x > 0 for x in excess_values])),
        "subwindow_mean_positive_excess_rate": float(np.mean(positive_values)) if positive_values else None,
    }


def _selection_score(metrics: dict, stability: dict, std_penalty: float, positive_bonus: float) -> float:
    mean_excess = float(metrics.get("mean_excess_return", float("-inf")))
    positive_rate = float(metrics.get("positive_excess_rate", 0.0))
    std_excess = stability.get("subwindow_std_excess")
    std_excess = float(std_excess) if std_excess is not None else 0.0
    return mean_excess - std_penalty * std_excess + positive_bonus * (positive_rate - 0.5)


def select_ensemble_and_portfolio(
    train_df,
    val_df,
    index_df,
    configs,
    top_ks,
    methods,
    lookbacks,
    stability_window_days,
    std_penalty,
    positive_bonus,
):
    leaderboard = []
    best_key = None
    best = None

    for config in configs:
        models, model_meta = _fit_recent_models(train_df, config, lookbacks)
        if models is None:
            leaderboard.append({"config_name": config["name"], "skipped": True, "model_windows": model_meta})
            continue
        for top_k in top_ks:
            for method in methods:
                metrics = evaluate_ensemble(models, val_df, index_df, top_k=top_k, weight_method=method)
                stability = _validation_subwindow_metrics(
                    models,
                    val_df,
                    index_df,
                    top_k=top_k,
                    weight_method=method,
                    subwindow_days=stability_window_days,
                )
                score = _selection_score(metrics, stability, std_penalty=std_penalty, positive_bonus=positive_bonus)
                row = {
                    "config_name": config["name"],
                "lookbacks": [m["lookback"] for m in model_meta if not m.get("skipped")],
                    "model_windows": model_meta,
                    "top_k": int(top_k),
                    "weight_method": method,
                    "stability_score": float(score),
                    **stability,
                    **metrics,
                }
                leaderboard.append(row)
                key = (
                    score,
                    metrics.get("mean_excess_return", float("-inf")),
                    metrics.get("positive_excess_rate", float("-inf")),
                    metrics.get("rank_ic", float("-inf")),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = (models, config["name"], model_meta, int(top_k), method, metrics, stability, float(score))

    if best is None:
        raise RuntimeError("No ensemble candidates were trainable; check lookbacks and data length.")
    return (*best, leaderboard)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument(
        "--weight-methods",
        default="softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.5_0.25,softmax_risk_1.5_0.50",
    )
    p.add_argument("--lookbacks", default="126,189,252")
    p.add_argument("--stability-window-days", type=int, default=5)
    p.add_argument("--std-penalty", type=float, default=0.5)
    p.add_argument("--positive-bonus", type=float, default=0.0025)
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

    print(">> Building recent-window ensemble feature panel")
    panel = build_features(prices, index_df)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df, val_df = splits["train_df"], splits["val_df"]
    print(f"   train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()}")

    print(">> Training recent-window rank ensemble")
    (
        models,
        best_config,
        model_meta,
        best_top_k,
        best_method,
        val_metrics,
        stability,
        stability_score,
        leaderboard,
    ) = select_ensemble_and_portfolio(
        train_df,
        val_df,
        index_df,
        MLP_CONFIGS,
        top_ks,
        methods,
        lookbacks,
        args.stability_window_days,
        args.std_penalty,
        args.positive_bonus,
    )
    print(f"   selected config/top_k/weight: {best_config} / {best_top_k} / {best_method}")
    print(f"   selected lookbacks: {','.join(lookbacks)}")
    print(f"   stability score: {stability_score:.6f}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in val_metrics:
        print(
            "   validation mean 5d returns "
            f"(portfolio/benchmark/excess): "
            f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{val_metrics['mean_excess_return']*100:+.3f}%"
        )
        print(
            f"   validation stability std excess: "
            f"{(stability.get('subwindow_std_excess') or 0)*100:.3f}%"
        )

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of).copy()
    pred_date = pred_df["date"].iloc[0]
    pred_df["score"] = predict_rank_ensemble(models, pred_df)
    weights = build_portfolio_custom(pred_df, top_k=best_top_k, method=best_method)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if args.json_out:
        payload = {
            "selected_config": best_config,
            "selected_top_k": best_top_k,
            "selected_weight_method": best_method,
            "selected_lookbacks": lookbacks,
            "model_windows": model_meta,
            "stability": stability,
            "stability_score": stability_score,
            "model_configs": MLP_CONFIGS,
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "stability_window_days": args.stability_window_days,
            "std_penalty": args.std_penalty,
            "positive_bonus": args.positive_bonus,
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
