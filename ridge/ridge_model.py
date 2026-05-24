"""
Ridge-regression model for the CSI500 stock-selection competition.

This model keeps the same data pipeline, split logic, and portfolio
construction as the XGBoost baseline so results are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import (  # noqa: E402
    DEFAULT_TOP_K,
    EMBARGO_DAYS,
    VAL_DAYS,
    build_portfolio,
    rank_ic,
)
from baseline.features import (  # noqa: E402
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
from baseline.paths import DATA_DIR  # noqa: E402


DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def build_dev_split(panel: pd.DataFrame, as_of: str | None = None) -> dict[str, pd.DataFrame | pd.Timestamp]:
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
    train_df = train_pool[train_pool["date"] <= train_end].copy()
    val_df = train_pool[train_pool["date"] >= val_start].copy()

    return {
        "train_df": train_df,
        "val_df": val_df,
        "train_end": train_end,
        "val_start": val_start,
        "train_cutoff": train_cutoff,
    }


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def period_excess_return(
    frame: pd.DataFrame,
    pred: np.ndarray,
    index_df: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    scored = frame[["date", "stock_code", TARGET_COLUMN]].copy()
    scored["score"] = pred

    index_panel = index_df.sort_values("date").copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel["bench_target"] = (
        index_panel["close"].shift(-FORWARD_HORIZON) / index_panel["close"] - 1.0
    )
    bench_fwd = index_panel.set_index("date")["bench_target"]

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

    result = pd.DataFrame(rows).sort_values("date")
    if result.empty:
        return result, None

    summary = {
        "n_dates": float(len(result)),
        "mean_portfolio_return": float(result["portfolio_return"].mean()),
        "mean_benchmark_return": float(result["benchmark_return"].mean()),
        "mean_excess_return": float(result["excess_return"].mean()),
        "positive_excess_rate": float((result["excess_return"] > 0).mean()),
    }
    return result, summary


def evaluate_model(
    model: Pipeline,
    frame: pd.DataFrame,
    index_df: pd.DataFrame,
    top_k: int,
) -> dict[str, float]:
    pred = model.predict(frame[FEATURE_COLUMNS])
    ic = rank_ic(frame[TARGET_COLUMN].to_numpy(), pred, frame["date"].to_numpy())
    _, bt = period_excess_return(frame, pred, index_df, top_k=top_k)
    out = {"rank_ic": float(ic)}
    if bt is not None:
        out.update(bt)
    return out


def select_alpha(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    index_df: pd.DataFrame,
    alphas: list[float],
    top_k: int,
) -> tuple[Pipeline, float, dict[str, float], list[dict[str, float]]]:
    leaderboard = []
    best_model = None
    best_alpha = None
    best_metrics = None
    best_key = None

    for alpha in alphas:
        model = make_model(alpha)
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        metrics = evaluate_model(model, val_df, index_df, top_k=top_k)
        leaderboard.append({"alpha": float(alpha), **metrics})

        key = (
            metrics.get("mean_excess_return", float("-inf")),
            metrics.get("rank_ic", float("-inf")),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_alpha = float(alpha)
            best_metrics = metrics

    assert best_model is not None and best_alpha is not None and best_metrics is not None
    return best_model, best_alpha, best_metrics, leaderboard


def parse_alphas(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--alphas", default="0.01,0.1,1,10,100")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None, help="Optional path to write metrics JSON")
    args = p.parse_args()

    alphas = parse_alphas(args.alphas)
    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )
    index_df = pd.read_parquet(args.index)

    print(">> Building features")
    panel = build_features(prices)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df = splits["train_df"]
    val_df = splits["val_df"]
    print(f"   train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()}")

    print(">> Training Ridge")
    model, best_alpha, val_metrics, leaderboard = select_alpha(
        train_df, val_df, index_df, alphas, top_k=args.top_k
    )
    print(f"   selected alpha: {best_alpha:g}")
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
            f"   validation positive excess rate: {val_metrics['positive_excess_rate']*100:.1f}% "
            f"over {int(val_metrics['n_dates'])} dates"
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
    print(
        f"   weight summary: min={out['weight'].min():.4f} "
        f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}"
    )

    if args.json_out:
        payload = {
            "selected_alpha": best_alpha,
            "alphas": alphas,
            "leaderboard": leaderboard,
            "validation": val_metrics,
            "prediction_date": pred_date.date().isoformat(),
            "top_k": args.top_k,
        }
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f">> Wrote Ridge summary to {json_path}")


if __name__ == "__main__":
    main()
