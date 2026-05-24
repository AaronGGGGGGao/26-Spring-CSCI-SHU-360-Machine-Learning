"""
Stage1-oriented 3-day MLP model.

This is a genuine non-tree alternative to test whether the current 3-day
XGBoost leader is only winning because of model family choice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

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

VAL_DAYS = 10
DEFAULT_TOP_KS = [30, 35, 40, 50]
DEFAULT_WEIGHT_METHODS = ["equal", "rank", "softmax", "score", "score_sq", "score_inv_vol", "rank_inv_vol"]
MLP_CONFIGS = [
    {
        "name": "small_relu",
        "hidden_layer_sizes": (64,),
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
    },
    {
        "name": "medium_relu",
        "hidden_layer_sizes": (128, 64),
        "alpha": 1e-3,
        "learning_rate_init": 8e-4,
    },
    {
        "name": "narrow_deep",
        "hidden_layer_sizes": (64, 32),
        "alpha": 1e-2,
        "learning_rate_init": 1e-3,
    },
]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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


def make_model(config: dict) -> Pipeline:
    mlp = MLPRegressor(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        alpha=config["alpha"],
        learning_rate_init=config["learning_rate_init"],
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        max_iter=300,
        random_state=42,
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def _cap_and_redistribute(w: pd.Series) -> pd.Series:
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
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def build_portfolio_custom(daily: pd.DataFrame, top_k: int, method: str) -> pd.Series:
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

    if method == "equal":
        w = pd.Series(1.0 / top_k, index=chosen.index)
    elif method == "rank":
        ranks = np.arange(top_k, 0, -1, dtype=float)
        w = pd.Series(ranks / ranks.sum(), index=chosen.index)
    elif method == "softmax":
        z = chosen["score"].to_numpy(dtype=float)
        z = (z - z.mean()) / (z.std() + 1e-12)
        e = np.exp(np.clip(1.5 * z, -20, 20))
        w = pd.Series(e / e.sum(), index=chosen.index)
    elif method == "score":
        z = chosen["score"] - chosen["score"].min() + 1e-8
        w = z / z.sum()
    elif method == "score_sq":
        z = chosen["score"] - chosen["score"].min() + 1e-8
        raw = np.square(z)
        w = raw / raw.sum()
    elif method == "score_inv_vol":
        z = chosen["score"] - chosen["score"].min() + 1e-8
        raw = z / risk
        w = raw / raw.sum()
    elif method == "rank_inv_vol":
        ranks = pd.Series(np.arange(top_k, 0, -1, dtype=float), index=chosen.index)
        raw = ranks / risk
        w = raw / raw.sum()
    else:
        raise ValueError(f"unknown weight method: {method}")
    return _cap_and_redistribute(w.astype(float))


def period_excess_return(frame: pd.DataFrame, pred: np.ndarray, index_df: pd.DataFrame, top_k: int, weight_method: str):
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
        weights = build_portfolio_custom(daily, top_k=top_k, method=weight_method)
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


def evaluate_model(model: Pipeline, frame: pd.DataFrame, index_df: pd.DataFrame, top_k: int, weight_method: str) -> dict[str, float]:
    pred = model.predict(frame[FEATURE_COLUMNS])
    ic = rank_ic(frame[TARGET_COLUMN].to_numpy(), pred, frame["date"].to_numpy())
    _, bt = period_excess_return(frame, pred, index_df, top_k=top_k, weight_method=weight_method)
    result = {"rank_ic": float(ic)}
    if bt is not None:
        result.update(bt)
    return result


def select_model_and_portfolio(train_df, val_df, index_df, configs, top_ks, methods):
    leaderboard = []
    best_key = None
    best = None
    for config in configs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = make_model(config)
            model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        for top_k in top_ks:
            for method in methods:
                metrics = evaluate_model(model, val_df, index_df, top_k=top_k, weight_method=method)
                row = {
                    "config_name": config["name"],
                    "top_k": int(top_k),
                    "weight_method": method,
                    **metrics,
                }
                leaderboard.append(row)
                key = (
                    metrics.get("mean_excess_return", float("-inf")),
                    metrics.get("positive_excess_rate", float("-inf")),
                    metrics.get("rank_ic", float("-inf")),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = (model, config["name"], int(top_k), method, metrics)
    assert best is not None
    return (*best, leaderboard)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35,40,50")
    p.add_argument("--weight-methods", default="equal,rank,softmax,score,score_sq,score_inv_vol,rank_inv_vol")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building feature panel")
    panel = build_features(prices, index_df)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df, val_df = splits["train_df"], splits["val_df"]
    print(f"   train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()}")

    print(">> Training MLP")
    model, best_config, best_top_k, best_method, val_metrics, leaderboard = select_model_and_portfolio(
        train_df, val_df, index_df, MLP_CONFIGS, top_ks, methods
    )
    print(f"   selected config/top_k/weight: {best_config} / {best_top_k} / {best_method}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in val_metrics:
        print(
            "   validation mean 5d returns "
            f"(portfolio/benchmark/excess): "
            f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{val_metrics['mean_excess_return']*100:+.3f}%"
        )
        print(f"   validation positive excess rate: {val_metrics['positive_excess_rate']*100:.1f}% over {int(val_metrics['n_dates'])} dates")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of).copy()
    pred_date = pred_df["date"].iloc[0]
    pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS])
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
            "model_configs": MLP_CONFIGS,
            "top_ks": top_ks,
            "weight_methods": methods,
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
