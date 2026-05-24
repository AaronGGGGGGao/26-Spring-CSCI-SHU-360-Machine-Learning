"""
Stage1 tuned MLP with covariance-aware constrained portfolio optimization.

Model side:
  - identical to the current tuned MLP leader

Portfolio side:
  - replace top-k + softmax weighting with
    alpha + shrunk covariance + teacher constraints
  - objective: maximize alpha^T w - lambda * w^T Sigma w
  - constraints:
      sum(w) = 1
      0 <= w_i <= 0.10
      at least 30 positive weights

Implementation choice:
  - choose a top-k candidate set by score
  - estimate a shrunk covariance matrix from trailing daily returns
  - solve constrained optimization with SLSQP
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import ConvergenceWarning
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS, MAX_WEIGHT, MIN_STOCKS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
from mlp.day5_stage2_tuned.mlp_model import (  # noqa: E402
    MLP_CONFIGS,
    make_model,
)
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
DEFAULT_TOP_KS = [30, 35, 40]
DEFAULT_COV_LOOKBACKS = [20, 40, 60]
DEFAULT_RISK_AVERSIONS = [0.5, 1.0, 2.0, 5.0]
MIN_ACTIVE_WEIGHT = 1e-4


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


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


def train_model(train_df: pd.DataFrame, config: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = make_model(config)
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return model


def build_returns_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    if "ret_1d" not in panel.columns:
        raise ValueError("feature panel is missing ret_1d required for covariance estimation")
    mat = panel.pivot(index="date", columns="stock_code", values="ret_1d").sort_index()
    return mat


def estimate_covariance(
    returns_matrix: pd.DataFrame,
    as_of_date: pd.Timestamp,
    names: list[str],
    lookback: int,
) -> np.ndarray:
    hist = returns_matrix.loc[returns_matrix.index < pd.Timestamp(as_of_date), names].tail(lookback)
    if hist.empty:
        return np.eye(len(names)) * 1e-4

    # Require at least a few observations; otherwise fallback to a diagonal matrix.
    hist = hist.copy()
    valid_rows = hist.dropna(how="all")
    if len(valid_rows) < 5:
        diag = np.nan_to_num(hist.var(axis=0).to_numpy(dtype=float), nan=1e-4, posinf=1e-4, neginf=1e-4)
        diag = np.where(diag <= 0, 1e-4, diag)
        return np.diag(diag)

    X = hist.fillna(0.0).to_numpy(dtype=float)
    lw = LedoitWolf().fit(X)
    cov = lw.covariance_.astype(float)
    cov = cov + np.eye(cov.shape[0]) * 1e-8
    return cov


def optimize_weights(
    alpha: pd.Series,
    covariance: np.ndarray,
    risk_aversion: float,
    *,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = MIN_ACTIVE_WEIGHT,
) -> pd.Series:
    n = len(alpha)
    if n < MIN_STOCKS:
        raise ValueError(f"need at least {MIN_STOCKS} names, got {n}")
    if min_weight * n >= 1.0:
        raise ValueError("min_weight is too large for the selected top_k")

    a = alpha.to_numpy(dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    cov = np.asarray(covariance, dtype=float)

    def objective(w: np.ndarray) -> float:
        return float(-(a @ w) + risk_aversion * (w @ cov @ w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight) for _ in range(n)]
    x0 = np.full(n, 1.0 / n, dtype=float)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )

    if not result.success or np.any(np.isnan(result.x)):
        # Robust fallback: score-proportional long-only weights with cap.
        raw = np.clip(a - a.min() + 1e-8, 1e-8, None)
        w = raw / raw.sum()
    else:
        w = result.x

    w = np.clip(w, min_weight, max_weight)
    w = w / w.sum()

    # Final projection loop to ensure exact feasibility after numeric drift.
    for _ in range(50):
        over = w > max_weight
        under = w < min_weight
        if not over.any() and not under.any():
            break
        w[over] = max_weight
        w[under] = min_weight
        free = (~over) & (~under)
        residual = 1.0 - w.sum()
        if free.any():
            w[free] += residual * w[free] / w[free].sum()
        else:
            w = w / w.sum()
            break

    w = w / w.sum()
    return pd.Series(w, index=alpha.index)


def build_portfolio_optimized(
    daily: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    *,
    top_k: int,
    cov_lookback: int,
    risk_aversion: float,
) -> pd.Series:
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")

    chosen = daily.sort_values("score", ascending=False).head(top_k).copy()
    if "stock_code" not in chosen.columns:
        chosen = chosen.reset_index()
    chosen = chosen.set_index("stock_code")

    names = chosen.index.tolist()
    cov = estimate_covariance(returns_matrix, pd.Timestamp(chosen["date"].iloc[0]), names, cov_lookback)
    weights = optimize_weights(chosen["score"], cov, risk_aversion)
    return weights


def period_excess_return(
    frame: pd.DataFrame,
    pred: np.ndarray,
    index_df: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    *,
    top_k: int,
    cov_lookback: int,
    risk_aversion: float,
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
        weights = build_portfolio_optimized(
            daily,
            returns_matrix,
            top_k=top_k,
            cov_lookback=cov_lookback,
            risk_aversion=risk_aversion,
        )
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


def evaluate_model(
    model,
    frame: pd.DataFrame,
    index_df: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    *,
    top_k: int,
    cov_lookback: int,
    risk_aversion: float,
) -> dict[str, float]:
    pred = model.predict(frame[FEATURE_COLUMNS])
    ic = rank_ic(frame[TARGET_COLUMN].to_numpy(), pred, frame["date"].to_numpy())
    _, bt = period_excess_return(
        frame,
        pred,
        index_df,
        returns_matrix,
        top_k=top_k,
        cov_lookback=cov_lookback,
        risk_aversion=risk_aversion,
    )
    result = {"rank_ic": float(ic)}
    if bt is not None:
        result.update(bt)
    return result


def select_model_and_portfolio(
    train_df,
    val_df,
    index_df,
    returns_matrix,
    configs,
    top_ks,
    cov_lookbacks,
    risk_aversions,
):
    leaderboard = []
    best_key = None
    best = None
    for config in configs:
        model = train_model(train_df, config)
        for top_k in top_ks:
            for cov_lookback in cov_lookbacks:
                for risk_aversion in risk_aversions:
                    metrics = evaluate_model(
                        model,
                        val_df,
                        index_df,
                        returns_matrix,
                        top_k=top_k,
                        cov_lookback=cov_lookback,
                        risk_aversion=risk_aversion,
                    )
                    row = {
                        "config_name": config["name"],
                        "top_k": int(top_k),
                        "cov_lookback": int(cov_lookback),
                        "risk_aversion": float(risk_aversion),
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
                        best = (model, config["name"], int(top_k), int(cov_lookback), float(risk_aversion), metrics)
    assert best is not None
    return (*best, leaderboard)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument("--cov-lookbacks", default="20,40,60")
    p.add_argument("--risk-aversions", default="0.5,1.0,2.0,5.0")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    cov_lookbacks = parse_int_list(args.cov_lookbacks)
    risk_aversions = parse_float_list(args.risk_aversions)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building feature panel")
    panel = build_features(prices, index_df)
    returns_matrix = build_returns_matrix(panel)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df, val_df = splits["train_df"], splits["val_df"]
    print(f"   train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()}")

    print(">> Training tuned MLP with optimized portfolio layer")
    model, best_config, best_top_k, best_cov_lookback, best_risk_aversion, val_metrics, leaderboard = (
        select_model_and_portfolio(
            train_df,
            val_df,
            index_df,
            returns_matrix,
            MLP_CONFIGS,
            top_ks,
            cov_lookbacks,
            risk_aversions,
        )
    )
    print(
        f"   selected config/top_k/cov_lookback/risk_aversion: "
        f"{best_config} / {best_top_k} / {best_cov_lookback} / {best_risk_aversion:.2f}"
    )
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
            f"   validation positive excess rate: "
            f"{val_metrics['positive_excess_rate']*100:.1f}% over {int(val_metrics['n_dates'])} dates"
        )

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of).copy()
    pred_date = pred_df["date"].iloc[0]
    pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS])
    weights = build_portfolio_optimized(
        pred_df,
        returns_matrix,
        top_k=best_top_k,
        cov_lookback=best_cov_lookback,
        risk_aversion=best_risk_aversion,
    )
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
            "selected_cov_lookback": best_cov_lookback,
            "selected_risk_aversion": best_risk_aversion,
            "model_configs": MLP_CONFIGS,
            "top_ks": top_ks,
            "cov_lookbacks": cov_lookbacks,
            "risk_aversions": risk_aversions,
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
