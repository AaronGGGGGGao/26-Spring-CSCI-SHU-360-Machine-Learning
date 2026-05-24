"""
XGBoost baseline for the CSI500 stock-selection competition.

Pipeline
--------
1. Load data/prices.parquet
2. Build features + 5-day forward target (features.py)
3. Train XGBoost on all but the last `EMBARGO_DAYS` training rows
4. Validate on those held-out rows (reports rank IC as sanity check)
5. Predict on the most recent date
6. Build a portfolio: top-K names, score-weighted with the 10% cap

Usage
-----
  python baseline_xgboost.py                       # predict from latest data
  python baseline_xgboost.py --as-of 20260503      # predict as of a given date
  python baseline_xgboost.py --top-k 50 --out submissions/week1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

try:
    from .features import (
        FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
        build_features, training_frame, prediction_frame,
    )
    from .paths import DATA_DIR
except ImportError:
    from features import (
        FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
        build_features, training_frame, prediction_frame,
    )
    from paths import DATA_DIR

VAL_DAYS = 10               # number of trading days in the validation window
EMBARGO_DAYS = 5            # gap between train end and val start (>= FORWARD_HORIZON
                            # so training targets don't reach into val dates)
MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap
DEFAULT_TOP_K = 50          # baseline picks top-50 by predicted score


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
    """Daily cross-sectional Spearman correlation, averaged over dates."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def validation_excess_return(
    val_df: pd.DataFrame,
    val_pred: np.ndarray,
    index_df: pd.DataFrame,
    top_k: int,
) -> dict[str, float] | None:
    """Held-out validation backtest using the model's daily predicted portfolios.

    For each validation date t:
      - build a portfolio from that day's cross-sectional scores
      - score it on the same 5-day forward horizon used by TARGET_COLUMN
      - compare against the CSI500 index's 5-day forward return from t to t+5

    These windows overlap across consecutive validation dates, so treat this as
    a model-selection sanity check rather than the competition's final metric.
    """
    scored = val_df[["date", "stock_code", TARGET_COLUMN]].copy()
    scored["score"] = val_pred

    index_panel = index_df.sort_values("date").copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel["bench_target_5d"] = (
        index_panel["close"].shift(-FORWARD_HORIZON) / index_panel["close"] - 1.0
    )
    bench_fwd = index_panel.set_index("date")["bench_target_5d"]

    rows = []
    for d, daily in scored.groupby("date"):
        bench_return = bench_fwd.get(pd.Timestamp(d))
        if pd.isna(bench_return):
            continue
        scores = daily.set_index("stock_code")["score"]
        weights = build_portfolio(scores, top_k=top_k)
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

    if not rows:
        return None

    result = pd.DataFrame(rows).sort_values("date")
    return {
        "n_dates": float(len(result)),
        "mean_portfolio_return": float(result["portfolio_return"].mean()),
        "mean_benchmark_return": float(result["benchmark_return"].mean()),
        "mean_excess_return": float(result["excess_return"].mean()),
        "positive_excess_rate": float((result["excess_return"] > 0).mean()),
    }


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    """Top-K names, weight proportional to (rank) then capped at MAX_WEIGHT.

    We use rank-weights rather than score-weights so pathological score scales
    do not produce a single dominant name.  After capping at 10% we redistribute
    spillover to uncapped names and iterate until feasible.
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    # Rank-based weights (best stock gets largest weight, then normalize).
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # Iteratively cap at MAX_WEIGHT and redistribute to uncapped names.
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

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")
    index_df = pd.read_parquet(args.index)

    print(">> Building features")
    panel = build_features(prices)
    # Bound training data so backtesting with --as-of doesn't leak future rows.
    # Training uses features from date t with target = close(t+FORWARD_HORIZON),
    # so we cap training dates at as_of - FORWARD_HORIZON trading days.
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    # Time-based split with embargo:
    #   [ ... train ... | embargo (discarded) | val (last VAL_DAYS) ]
    # The embargo prevents training labels (5-day forward) from reaching into
    # dates whose prices also feed the validation features.
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
    val_bt = validation_excess_return(val_df, val_pred, index_df, top_k=args.top_k)
    if val_bt is not None:
        print(
            "   validation mean 5d returns "
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
    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={args.as_of}. Check data.")
    pred_date = pred_df["date"].iloc[0]
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} "
          f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
