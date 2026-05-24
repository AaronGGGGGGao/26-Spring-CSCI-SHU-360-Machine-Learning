"""
Recent-window MLP with a richer portfolio-construction search.

This branch keeps the current best recent-window score model family and focuses
on the part that still looks unstable: mapping scores to weights. It expands the
weight-rule search to include bucketed, equal-weight, and clipped-linear
variants, then selects configurations with a penalty for unstable validation
subwindow excess returns.
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

from baseline.baseline_xgboost import EMBARGO_DAYS, MAX_WEIGHT, MIN_STOCKS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
try:  # noqa: E402
    from .features import (
        FEATURE_COLUMNS,
        FORWARD_HORIZON,
        TARGET_COLUMN,
        build_features,
        prediction_frame,
        training_frame,
    )
except ImportError:  # noqa: E402
    from features import (
        FEATURE_COLUMNS,
        FORWARD_HORIZON,
        TARGET_COLUMN,
        build_features,
        prediction_frame,
        training_frame,
    )

from mlp.day3_stage1_tuned.mlp_model import (  # noqa: E402
    MLP_CONFIGS,
    _softmax_weights,
    make_model,
    parse_int_list,
    parse_str_list,
    rank_ic,
)

VAL_DAYS = 10
DEFAULT_LOOKBACKS = ["63", "126", "189", "252", "full"]
DEFAULT_TOP_KS = [30, 35, 40, 45]
DEFAULT_WEIGHT_METHODS = [
    "softmax_1.8",
    "softmax_2.0",
    "softmax_risk_1.5_0.25",
    "softmax_risk_1.5_0.50",
    "equal",
    "clipped_linear",
    "clipped_linear_risk_0.25",
    "bucket_40_35_25",
    "bucket_45_35_20",
]


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


def _cap_and_redistribute(w: pd.Series) -> pd.Series:
    w = w.astype(float)
    if (w < 0).any():
        w = w.clip(lower=0.0)
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / w.sum()
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        free_sum = w[free].sum()
        if free_sum <= 0:
            w[free] = 1.0 / free.sum()
            free_sum = 1.0
        w[free] += excess * w[free] / free_sum
    return w / w.sum()


def _prepare_chosen(daily: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, pd.Series]:
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
    chosen = daily.sort_values("score", ascending=False).head(top_k).copy()
    if "stock_code" not in chosen.columns:
        chosen = chosen.reset_index()
    chosen = chosen.set_index("stock_code")
    risk = chosen["idio_vol_20d"].replace(0, np.nan)
    fallback_risk = risk.median()
    if pd.isna(fallback_risk) or fallback_risk <= 0:
        fallback_risk = 1.0
    risk = risk.fillna(fallback_risk)
    return chosen, risk


def _bucket_weights(index: pd.Index, shares: list[float]) -> pd.Series:
    n = len(index)
    if n == 0:
        return pd.Series(dtype=float)
    bucket_sizes = [n // 3, n // 3, n - 2 * (n // 3)]
    weights = np.zeros(n, dtype=float)
    start = 0
    for size, share in zip(bucket_sizes, shares):
        if size <= 0:
            continue
        end = start + size
        weights[start:end] = share / size
        start = end
    return pd.Series(weights, index=index)


def build_portfolio_weight_optimized(daily: pd.DataFrame, top_k: int, method: str) -> pd.Series:
    chosen, risk = _prepare_chosen(daily, top_k)
    if method.startswith("softmax_risk_"):
        _, _, temp_str, power_str = method.split("_")
        temperature = float(temp_str)
        power = float(power_str)
        base = _softmax_weights(chosen["score"], temperature)
        raw = base / np.power(risk, power)
        w = raw / raw.sum()
    elif method.startswith("softmax_"):
        temperature = float(method.split("_", 1)[1])
        w = _softmax_weights(chosen["score"], temperature)
    elif method == "equal":
        w = pd.Series(np.full(len(chosen), 1.0 / len(chosen)), index=chosen.index)
    elif method == "clipped_linear":
        ranks = chosen["score"].rank(method="first", ascending=False)
        linear = (len(chosen) + 1 - ranks).astype(float)
        w = linear / linear.sum()
    elif method == "clipped_linear_risk_0.25":
        ranks = chosen["score"].rank(method="first", ascending=False)
        linear = (len(chosen) + 1 - ranks).astype(float)
        raw = linear / np.power(risk, 0.25)
        w = raw / raw.sum()
    elif method == "bucket_40_35_25":
        w = _bucket_weights(chosen.index, [0.40, 0.35, 0.25])
    elif method == "bucket_45_35_20":
        w = _bucket_weights(chosen.index, [0.45, 0.35, 0.20])
    else:
        raise ValueError(f"unknown weight method: {method}")
    return _cap_and_redistribute(w)


def period_excess_return_weight_optimized(
    frame: pd.DataFrame,
    pred: np.ndarray,
    index_df: pd.DataFrame,
    top_k: int,
    weight_method: str,
):
    scored = frame.copy()
    scored["score"] = pred

    index_panel = index_df.sort_values("date").copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel["bench_target"] = index_panel["close"].shift(-FORWARD_HORIZON) / index_panel["close"] - 1.0
    bench_fwd = index_panel.set_index("date")["bench_target"]

    rows = []
    for d, daily in scored.groupby("date"):
        bench_return = bench_fwd.get(pd.Timestamp(d))
        if pd.isna(bench_return):
            continue
        weights = build_portfolio_weight_optimized(daily, top_k=top_k, method=weight_method)
        realized = daily.set_index("stock_code")[TARGET_COLUMN].reindex(weights.index)
        portfolio_return = float((weights * realized).sum())
        rows.append(
            {
                "date": pd.Timestamp(d),
                "portfolio_return": portfolio_return,
                "benchmark_return": float(bench_return),
                "excess_return": portfolio_return - float(bench_return),
            }
        )

    result = pd.DataFrame(rows).sort_values("date")
    if result.empty:
        return result, None
    return result, {
        "n_dates": float(len(result)),
        "mean_portfolio_return": float(result["portfolio_return"].mean()),
        "mean_benchmark_return": float(result["benchmark_return"].mean()),
        "mean_excess_return": float(result["excess_return"].mean()),
        "positive_excess_rate": float((result["excess_return"] > 0).mean()),
    }


def evaluate_model_weight_optimized(
    model,
    frame: pd.DataFrame,
    index_df: pd.DataFrame,
    top_k: int,
    weight_method: str,
) -> dict[str, float]:
    pred = model.predict(frame[FEATURE_COLUMNS])
    ic = rank_ic(frame[TARGET_COLUMN].to_numpy(), pred, frame["date"].to_numpy())
    _, bt = period_excess_return_weight_optimized(frame, pred, index_df, top_k=top_k, weight_method=weight_method)
    result = {"rank_ic": float(ic)}
    if bt is not None:
        result.update(bt)
    return result


def _validation_subwindow_metrics(
    model,
    val_df: pd.DataFrame,
    index_df: pd.DataFrame,
    top_k: int,
    weight_method: str,
    subwindow_days: int,
) -> dict[str, float | None]:
    dates = np.sort(val_df["date"].unique())
    chunks = [dates[i : i + subwindow_days] for i in range(0, len(dates), subwindow_days)]
    excess_values: list[float] = []
    positive_values: list[float] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        chunk_df = val_df[val_df["date"].isin(chunk)].copy()
        metrics = evaluate_model_weight_optimized(model, chunk_df, index_df, top_k=top_k, weight_method=weight_method)
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
        "subwindow_mean_positive_excess_rate": float(np.mean(positive_values)),
    }


def _selection_score(metrics: dict, stability: dict, std_penalty: float, positive_bonus: float) -> float:
    mean_excess = float(metrics.get("mean_excess_return", float("-inf")))
    positive_rate = float(metrics.get("positive_excess_rate", 0.0))
    std_excess = stability.get("subwindow_std_excess")
    std_excess = float(std_excess) if std_excess is not None else 0.0
    return mean_excess - std_penalty * std_excess + positive_bonus * (positive_rate - 0.5)


def select_with_lookback_and_weight_optimization(
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
        for config in configs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model = make_model(config)
                model.fit(recent_train[FEATURE_COLUMNS], recent_train[TARGET_COLUMN])
            for top_k in top_ks:
                for method in methods:
                    metrics = evaluate_model_weight_optimized(
                        model, val_df, index_df, top_k=top_k, weight_method=method
                    )
                    stability = _validation_subwindow_metrics(
                        model,
                        val_df,
                        index_df,
                        top_k=top_k,
                        weight_method=method,
                        subwindow_days=stability_window_days,
                    )
                    score = _selection_score(
                        metrics, stability, std_penalty=std_penalty, positive_bonus=positive_bonus
                    )
                    row = {
                        "lookback": lookback,
                        "train_window_start": start_marker,
                        "train_rows": int(len(recent_train)),
                        "train_dates": int(unique_dates),
                        "config_name": config["name"],
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
                        best = (
                            model,
                            lookback,
                            start_marker,
                            int(len(recent_train)),
                            int(unique_dates),
                            config["name"],
                            int(top_k),
                            method,
                            metrics,
                            stability,
                            float(score),
                        )
    if best is None:
        raise RuntimeError("All recent-window candidates were skipped; not enough training data.")
    return (*best, leaderboard)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35,40,45")
    p.add_argument(
        "--weight-methods",
        default="softmax_1.8,softmax_2.0,softmax_risk_1.5_0.25,softmax_risk_1.5_0.50,equal,clipped_linear,clipped_linear_risk_0.25,bucket_40_35_25,bucket_45_35_20",
    )
    p.add_argument("--lookbacks", default="63,126,189,252,full")
    p.add_argument("--stability-window-days", type=int, default=5)
    p.add_argument("--std-penalty", type=float, default=0.50)
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

    print(">> Building recent-window weight-optimized feature panel")
    panel = build_features(prices, index_df)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df, val_df = splits["train_df"], splits["val_df"]
    print(f"   base train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()}")

    print(">> Training recent-window weight-optimized MLP")
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
        stability,
        stability_score,
        leaderboard,
    ) = select_with_lookback_and_weight_optimization(
        train_df,
        val_df,
        index_df,
        MLP_CONFIGS,
        top_ks,
        methods,
        lookbacks,
        stability_window_days=args.stability_window_days,
        std_penalty=args.std_penalty,
        positive_bonus=args.positive_bonus,
    )
    print(
        "   selected lookback/config/top_k/weight: "
        f"{best_lookback} ({best_start_marker}) / {best_config} / {best_top_k} / {best_method}"
    )
    print(f"   selected recent train rows/dates: {best_train_rows:,} / {best_train_dates}")
    print(f"   validation stability score: {stability_score:.6f}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in val_metrics:
        print(
            "   validation mean 3d returns "
            f"(portfolio/benchmark/excess): "
            f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{val_metrics['mean_excess_return']*100:+.3f}%"
        )
    if stability.get("subwindow_std_excess") is not None:
        print(f"   validation stability std excess: {stability['subwindow_std_excess']*100:.3f}%")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of).copy()
    pred_date = pred_df["date"].iloc[0]
    pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS])
    weights = build_portfolio_weight_optimized(pred_df, top_k=best_top_k, method=best_method)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")
    print(f">> Wrote {len(out)} names to {out_path}")
    print(
        f"   weight summary: min={out['weight'].min():.4f} "
        f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}"
    )

    if args.json_out:
        payload = {
            "selected_lookback": best_lookback,
            "selected_train_window_start": best_start_marker,
            "selected_train_rows": best_train_rows,
            "selected_train_dates": best_train_dates,
            "selected_config": best_config,
            "selected_top_k": best_top_k,
            "selected_weight_method": best_method,
            "selected_stability_score": stability_score,
            "validation_stability": stability,
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
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f">> Wrote summary to {json_path}")


if __name__ == "__main__":
    main()
