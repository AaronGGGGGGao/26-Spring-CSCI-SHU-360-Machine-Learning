"""
3-day XGBoost baseline aligned to the stage1 holding horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import DEFAULT_TOP_K, EMBARGO_DAYS, MIN_STOCKS, MAX_WEIGHT
from baseline.paths import DATA_DIR
from baseline.day3.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FORWARD_HORIZON,
    build_features,
    training_frame,
    prediction_frame,
)

VAL_DAYS = 10


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


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


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)
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
    return w


def period_excess_return(frame: pd.DataFrame, pred: np.ndarray, index_df: pd.DataFrame, top_k: int):
    scored = frame[["date", "stock_code", TARGET_COLUMN]].copy()
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
        weights = build_portfolio(daily.set_index("stock_code")["score"], top_k=top_k)
        realized = daily.set_index("stock_code")[TARGET_COLUMN].reindex(weights.index)
        portfolio_return = float((weights * realized).sum())
        rows.append({
            "date": pd.Timestamp(d),
            "portfolio_return": portfolio_return,
            "benchmark_return": float(bench_return),
            "excess_return": portfolio_return - float(bench_return),
        })

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building features")
    panel = build_features(prices)
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
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
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]
    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")

    print(">> Training XGBoost")
    model = train_model(train_df, val_df)
    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   validation rank IC: {ic:.4f}")
    _, val_bt = period_excess_return(val_df, val_pred, index_df, top_k=args.top_k)
    if val_bt is not None:
        print(
            "   validation mean 3d returns "
            f"(portfolio/benchmark/excess): "
            f"{val_bt['mean_portfolio_return']*100:+.3f}% / "
            f"{val_bt['mean_benchmark_return']*100:+.3f}% / "
            f"{val_bt['mean_excess_return']*100:+.3f}%"
        )
        print(
            f"   validation positive excess rate: {val_bt['positive_excess_rate']*100:.1f}% "
            f"over {int(val_bt['n_dates'])} dates"
        )

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of)
    pred_date = pred_df["date"].iloc[0]
    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
