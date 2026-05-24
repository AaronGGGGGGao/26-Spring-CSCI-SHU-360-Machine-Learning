"""
Style-factor and dynamic-allocation enhanced 3-day tuned MLP.

Design:
  - keep the proven tuned MLP family as the alpha engine
  - add structured style/risk features derived from public market data
  - search only recency weighting, not unstable external-data features
  - upgrade score->weight from one static rule to a small set of
    regime-dependent allocation policies
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
RECENCY_HALF_LIVES = [10, 20, 40]
MLP_CONFIGS = [
    {
        "name": "narrow_deep",
        "hidden_layer_sizes": (64, 32),
        "alpha": 1e-2,
        "learning_rate_init": 1e-3,
    },
    {
        "name": "narrow_deep_lighter_reg",
        "hidden_layer_sizes": (64, 32),
        "alpha": 5e-3,
        "learning_rate_init": 1e-3,
    },
    {
        "name": "wider_deep",
        "hidden_layer_sizes": (96, 48),
        "alpha": 1e-2,
        "learning_rate_init": 8e-4,
    },
]
ALLOCATION_POLICIES = [
    {
        "name": "static_softmax_2.0",
        "bull": {"top_k": 30, "method": "softmax_2.0"},
        "neutral": {"top_k": 30, "method": "softmax_2.0"},
        "stress": {"top_k": 30, "method": "softmax_2.0"},
    },
    {
        "name": "trend_dynamic",
        "bull": {"top_k": 30, "method": "softmax_2.4"},
        "neutral": {"top_k": 30, "method": "softmax_1.8"},
        "stress": {"top_k": 35, "method": "softmax_risk_1.8_0.35"},
    },
    {
        "name": "breadth_dynamic",
        "bull": {"top_k": 30, "method": "score_sq"},
        "neutral": {"top_k": 30, "method": "softmax_2.0"},
        "stress": {"top_k": 35, "method": "softmax_risk_1.5_0.50"},
    },
    {
        "name": "defensive_dynamic",
        "bull": {"top_k": 30, "method": "softmax_2.0"},
        "neutral": {"top_k": 35, "method": "softmax_1.5"},
        "stress": {"top_k": 35, "method": "softmax_risk_2.0_0.35"},
    },
]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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


def _sample_weights(train_df: pd.DataFrame, half_life: int) -> np.ndarray:
    date_order = {d: i for i, d in enumerate(np.sort(train_df["date"].unique()))}
    last_idx = max(date_order.values())
    age = train_df["date"].map(lambda d: last_idx - date_order[pd.Timestamp(d)]).to_numpy(dtype=float)
    w = np.power(0.5, age / float(half_life))
    w = w / np.mean(w)
    return w.astype(float)


def _fit_model(config: dict, train_df: pd.DataFrame, sample_weight: np.ndarray) -> Pipeline:
    model = make_model(config)
    try:
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], mlp__sample_weight=sample_weight)
    except TypeError:
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return model


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


def _softmax_weights(scores: pd.Series, temperature: float) -> pd.Series:
    z = scores.to_numpy(dtype=float)
    z = (z - z.mean()) / (z.std() + 1e-12)
    e = np.exp(np.clip(temperature * z, -20, 20))
    return pd.Series(e / e.sum(), index=scores.index)


def build_portfolio_custom(daily: pd.DataFrame, top_k: int, method: str) -> pd.Series:
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
    chosen = daily.sort_values("score", ascending=False).head(top_k).copy()
    if "stock_code" in chosen.columns:
        chosen = chosen.set_index("stock_code")
    else:
        if chosen.index.name == "stock_code":
            pass
        elif "stock_code" in list(chosen.index.names):
            chosen = chosen.reset_index().set_index("stock_code")
        else:
            chosen = chosen.reset_index()
            index_name = chosen.columns[0]
            chosen = chosen.rename(columns={index_name: "stock_code"}).set_index("stock_code")

    risk = chosen["idio_vol_20d"].replace(0, np.nan)
    fallback_risk = risk.median()
    if pd.isna(fallback_risk) or fallback_risk <= 0:
        fallback_risk = 1.0
    risk = risk.fillna(fallback_risk)

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
    elif method == "score_sq":
        base = chosen["score"] - chosen["score"].min() + 1e-6
        raw = np.square(base)
        w = raw / raw.sum()
    else:
        raise ValueError(f"unknown weight method: {method}")
    return _cap_and_redistribute(w.astype(float))


def _market_state(daily: pd.DataFrame) -> str:
    trend = float(daily["mkt_trend_5_20"].median())
    breadth = float(daily["mkt_breadth_10d"].median())
    drawdown = float(daily["mkt_drawdown_20d"].median())
    vol_regime = float(daily["mkt_vol_regime"].median())
    if drawdown <= -0.07 or (vol_regime > 0.5 and breadth < 0.50) or trend < -0.01:
        return "stress"
    if trend >= 0.01 and breadth >= 0.52 and drawdown > -0.05:
        return "bull"
    return "neutral"


def _policy_choice(policy_name: str, daily: pd.DataFrame) -> tuple[int, str]:
    policy = next(p for p in ALLOCATION_POLICIES if p["name"] == policy_name)
    state = _market_state(daily)
    choice = policy[state]
    return int(choice["top_k"]), str(choice["method"])


def period_excess_return(frame: pd.DataFrame, pred: np.ndarray, index_df: pd.DataFrame, policy_name: str):
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
        top_k, method = _policy_choice(policy_name, daily)
        weights = build_portfolio_custom(daily, top_k=top_k, method=method)
        if "stock_code" in daily.columns:
            realized_source = daily.set_index("stock_code")
        else:
            if daily.index.name == "stock_code":
                realized_source = daily
            elif "stock_code" in list(daily.index.names):
                realized_source = daily.reset_index().set_index("stock_code")
            else:
                reset_daily = daily.reset_index()
                index_name = reset_daily.columns[0]
                realized_source = reset_daily.rename(columns={index_name: "stock_code"}).set_index("stock_code")
        realized = realized_source[TARGET_COLUMN].reindex(weights.index)
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


def evaluate_model(model: Pipeline, frame: pd.DataFrame, index_df: pd.DataFrame, policy_name: str) -> dict[str, float]:
    pred = model.predict(frame[FEATURE_COLUMNS])
    ic = rank_ic(frame[TARGET_COLUMN].to_numpy(), pred, frame["date"].to_numpy())
    _, bt = period_excess_return(frame, pred, index_df, policy_name=policy_name)
    result = {"rank_ic": float(ic)}
    if bt is not None:
        result.update(bt)
    return result


def select_model_and_policy(train_df, val_df, index_df, configs, half_lives, policy_names):
    leaderboard = []
    best_key = None
    best = None
    for config in configs:
        for half_life in half_lives:
            sample_weight = _sample_weights(train_df, half_life=half_life)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model = _fit_model(config, train_df, sample_weight=sample_weight)
            for policy_name in policy_names:
                metrics = evaluate_model(model, val_df, index_df, policy_name=policy_name)
                row = {
                    "config_name": config["name"],
                    "half_life": int(half_life),
                    "allocation_policy": policy_name,
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
                    best = (model, config["name"], int(half_life), policy_name, metrics)
    assert best is not None
    return (*best, leaderboard)


def _prediction_portfolio(pred_df: pd.DataFrame, policy_name: str) -> pd.Series:
    pred_df = pred_df.copy()
    top_k, method = _policy_choice(policy_name, pred_df)
    return build_portfolio_custom(pred_df, top_k=top_k, method=method)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--half-lives", default="10,20,40")
    p.add_argument("--policies", default="static_softmax_2.0,trend_dynamic,breadth_dynamic,defensive_dynamic")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    half_lives = parse_int_list(args.half_lives)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building style-dynamic feature panel")
    panel = build_features(prices, index_df)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df, val_df = splits["train_df"], splits["val_df"]
    print(f"   train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()}")

    print(">> Training style-dynamic tuned MLP")
    model, best_config, best_half_life, best_policy, val_metrics, leaderboard = select_model_and_policy(
        train_df, val_df, index_df, MLP_CONFIGS, half_lives, policies
    )
    print(f"   selected config/half_life/policy: {best_config} / {best_half_life} / {best_policy}")
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
    weights = _prediction_portfolio(pred_df, policy_name=best_policy)
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
            "selected_config": best_config,
            "selected_half_life": best_half_life,
            "selected_allocation_policy": best_policy,
            "model_configs": MLP_CONFIGS,
            "half_lives": half_lives,
            "policies": policies,
            "policy_definitions": ALLOCATION_POLICIES,
            "leaderboard": leaderboard,
            "validation": val_metrics,
            "prediction_date": pred_date.date().isoformat(),
        }
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f">> Wrote summary to {out_json}")


if __name__ == "__main__":
    main()
