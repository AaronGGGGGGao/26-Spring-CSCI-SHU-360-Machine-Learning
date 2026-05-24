"""
Recent-window MLP trained on 3-day excess return with recency sample weighting.

This branch keeps the current leader's features and portfolio construction, but
aligns the supervised target with the competition metric:

    target_excess_3d = stock forward 3d return - CSI500 forward 3d return
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

from mlp.day3_stage1_tuned.mlp_model import (  # noqa: E402
    MLP_CONFIGS,
    build_portfolio_custom,
    make_model,
    parse_int_list,
    parse_str_list,
    rank_ic,
)

VAL_DAYS = 10
DEFAULT_LOOKBACKS = ["126", "189"]
DEFAULT_HALF_LIVES = [20, 40]
DEFAULT_TOP_KS = [30, 35, 40]
DEFAULT_WEIGHT_METHODS = [
    "softmax_1.5",
    "softmax_1.8",
    "softmax_2.0",
    "softmax_risk_1.5_0.50",
]
DEFAULT_TARGET_TRANSFORMS = ["winsor_zscore", "rank"]
DEFAULT_POLICIES = [
    "static_30_softmax_2.0",
    "static_35_softmax_2.0",
    "static_40_softmax_risk_1.5_0.50",
    "regime_light",
    "regime_defensive",
]


def parse_lookbacks(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def parse_policies(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _restrict_recent_window(train_df: pd.DataFrame, lookback: str) -> tuple[pd.DataFrame, str]:
    if lookback == "full":
        return train_df.copy(), "full"
    lookback_days = int(lookback)
    train_dates = np.sort(train_df["date"].unique())
    if len(train_dates) < lookback_days:
        return train_df.copy(), "full_fallback"
    start_date = pd.Timestamp(train_dates[-lookback_days])
    return train_df[train_df["date"] >= start_date].copy(), start_date.date().isoformat()


def _sample_weights(train_df: pd.DataFrame, half_life: int) -> np.ndarray:
    date_order = {pd.Timestamp(d): i for i, d in enumerate(np.sort(train_df["date"].unique()))}
    last_idx = max(date_order.values())
    age = train_df["date"].map(lambda d: last_idx - date_order[pd.Timestamp(d)]).to_numpy(dtype=float)
    w = np.power(0.5, age / float(half_life))
    w = w / np.mean(w)
    return w.astype(float)


def _transformed_target(train_df: pd.DataFrame, transform: str) -> np.ndarray:
    if transform == "raw":
        return train_df[TARGET_COLUMN].to_numpy(dtype=float)

    def per_date(s: pd.Series) -> pd.Series:
        if transform == "winsor_zscore":
            lo = s.quantile(0.02)
            hi = s.quantile(0.98)
            clipped = s.clip(lo, hi)
            std = clipped.std()
            if pd.isna(std) or std <= 1e-12:
                return clipped * 0.0
            return (clipped - clipped.mean()) / std
        if transform == "rank":
            return s.rank(method="average", pct=True) - 0.5
        raise ValueError(f"unknown target transform: {transform}")

    return train_df.groupby("date", group_keys=False)[TARGET_COLUMN].transform(per_date).to_numpy(dtype=float)


def _tail_downweight(train_df: pd.DataFrame) -> np.ndarray:
    def per_date(s: pd.Series) -> pd.Series:
        lo = s.quantile(0.02)
        hi = s.quantile(0.98)
        return pd.Series(np.where((s < lo) | (s > hi), 0.5, 1.0), index=s.index)

    return train_df.groupby("date", group_keys=False)[TARGET_COLUMN].transform(per_date).to_numpy(dtype=float)


def _fit_excess_model(config: dict, train_df: pd.DataFrame, half_life: int, target_transform: str):
    model = make_model(config)
    y = _transformed_target(train_df, target_transform)
    sample_weight = _sample_weights(train_df, half_life=half_life) * _tail_downweight(train_df)
    sample_weight = sample_weight / np.mean(sample_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        try:
            model.fit(train_df[FEATURE_COLUMNS], y, mlp__sample_weight=sample_weight)
        except TypeError:
            model.fit(train_df[FEATURE_COLUMNS], y)
    return model


def _index_state(index_df: pd.DataFrame) -> pd.DataFrame:
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    close = idx["close"].astype(float)
    idx["mkt_ret_5d"] = close.pct_change(5)
    idx["mkt_ret_10d"] = close.pct_change(10)
    idx["mkt_ret_20d"] = close.pct_change(20)
    ret_1d = close.pct_change(1)
    idx["mkt_vol_10d"] = ret_1d.rolling(10).std()
    idx["mkt_vol_20d"] = ret_1d.rolling(20).std()
    idx["mkt_vol_regime"] = (idx["mkt_vol_10d"] > idx["mkt_vol_10d"].rolling(60).median()).astype(float)
    idx["mkt_trend_5_20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1.0
    idx["mkt_drawdown_20d"] = close / close.rolling(20).max() - 1.0
    return idx.set_index("date")


def _market_state(daily: pd.DataFrame, index_state: pd.DataFrame) -> str:
    d = pd.Timestamp(daily["date"].iloc[0])
    row = index_state.loc[d]
    breadth = float((daily["ret_1d"] > 0).mean()) if "ret_1d" in daily.columns else 0.5
    trend = float(row.get("mkt_trend_5_20", 0.0))
    drawdown = float(row.get("mkt_drawdown_20d", 0.0))
    vol_regime = float(row.get("mkt_vol_regime", 0.0))
    if drawdown <= -0.04 or (vol_regime > 0.5 and breadth < 0.50) or trend < -0.01:
        return "stress"
    if trend >= 0.01 and breadth >= 0.52 and drawdown > -0.04:
        return "bull"
    return "neutral"


def _policy_choice(policy_name: str, daily: pd.DataFrame, index_state: pd.DataFrame) -> tuple[int, str]:
    if policy_name.startswith("static_"):
        _, top_k, method = policy_name.split("_", 2)
        return int(top_k), method

    state = _market_state(daily, index_state)
    if policy_name == "regime_light":
        choices = {
            "bull": (30, "softmax_2.0"),
            "neutral": (35, "softmax_1.8"),
            "stress": (40, "softmax_risk_1.5_0.50"),
        }
    elif policy_name == "regime_defensive":
        choices = {
            "bull": (30, "softmax_2.0"),
            "neutral": (40, "softmax_risk_1.5_0.50"),
            "stress": (40, "softmax_risk_1.5_0.50"),
        }
    else:
        raise ValueError(f"unknown allocation policy: {policy_name}")
    return choices[state]


def period_excess_return_policy(frame: pd.DataFrame, pred: np.ndarray, index_df: pd.DataFrame, policy_name: str):
    scored = frame.copy()
    scored["score"] = pred

    index_panel = index_df.sort_values("date").copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel["bench_target"] = index_panel["close"].shift(-FORWARD_HORIZON) / index_panel["close"] - 1.0
    bench_fwd = index_panel.set_index("date")["bench_target"]
    index_state = _index_state(index_df)

    rows = []
    policy_rows = []
    for d, daily in scored.groupby("date"):
        bench_return = bench_fwd.get(pd.Timestamp(d))
        if pd.isna(bench_return):
            continue
        top_k, method = _policy_choice(policy_name, daily, index_state)
        weights = build_portfolio_custom(daily, top_k=top_k, method=method)
        realized = daily.set_index("stock_code")["target_3d"].reindex(weights.index)
        portfolio_return = float((weights * realized).sum())
        rows.append(
            {
                "date": pd.Timestamp(d),
                "portfolio_return": portfolio_return,
                "benchmark_return": float(bench_return),
                "excess_return": portfolio_return - float(bench_return),
            }
        )
        policy_rows.append({"date": pd.Timestamp(d), "top_k": int(top_k), "weight_method": method})

    result = pd.DataFrame(rows).sort_values("date")
    if result.empty:
        return result, None, pd.DataFrame(policy_rows)
    metrics = {
        "n_dates": float(len(result)),
        "mean_portfolio_return": float(result["portfolio_return"].mean()),
        "mean_benchmark_return": float(result["benchmark_return"].mean()),
        "mean_excess_return": float(result["excess_return"].mean()),
        "positive_excess_rate": float((result["excess_return"] > 0).mean()),
    }
    return result, metrics, pd.DataFrame(policy_rows)


def evaluate_model_policy(model, frame: pd.DataFrame, index_df: pd.DataFrame, policy_name: str) -> dict[str, float]:
    pred = model.predict(frame[FEATURE_COLUMNS])
    ic = rank_ic(frame[TARGET_COLUMN].to_numpy(), pred, frame["date"].to_numpy())
    _, bt, policy_rows = period_excess_return_policy(frame, pred, index_df, policy_name=policy_name)
    result = {"rank_ic": float(ic)}
    if bt is not None:
        result.update(bt)
    if not policy_rows.empty:
        result["mean_top_k"] = float(policy_rows["top_k"].mean())
    return result


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


def select_with_excess_target(
    train_df,
    val_df,
    index_df,
    configs,
    top_ks,
    methods,
    lookbacks,
    half_lives,
    target_transforms,
    policies,
):
    leaderboard = []
    best_key = None
    best = None
    static_policies = [f"static_{top_k}_{method}" for top_k in top_ks for method in methods]
    candidate_policies = static_policies + [p for p in policies if not p.startswith("static_")]
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

        for half_life in half_lives:
            for target_transform in target_transforms:
                for config in configs:
                    model = _fit_excess_model(
                        config,
                        recent_train,
                        half_life=half_life,
                        target_transform=target_transform,
                    )
                    for policy_name in candidate_policies:
                        metrics = evaluate_model_policy(model, val_df, index_df, policy_name=policy_name)
                        row = {
                            "lookback": lookback,
                            "train_window_start": start_marker,
                            "train_rows": int(len(recent_train)),
                            "train_dates": int(unique_dates),
                            "half_life": int(half_life),
                            "target_transform": target_transform,
                            "config_name": config["name"],
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
                            best = (
                                model,
                                lookback,
                                start_marker,
                                int(len(recent_train)),
                                int(unique_dates),
                                int(half_life),
                                target_transform,
                                config["name"],
                                policy_name,
                                metrics,
                            )
    if best is None:
        raise RuntimeError("All excess-target candidates were skipped; not enough training data.")
    return (*best, leaderboard)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default=",".join(str(x) for x in DEFAULT_TOP_KS))
    p.add_argument("--weight-methods", default=",".join(DEFAULT_WEIGHT_METHODS))
    p.add_argument("--lookbacks", default=",".join(DEFAULT_LOOKBACKS))
    p.add_argument("--half-lives", default=",".join(str(x) for x in DEFAULT_HALF_LIVES))
    p.add_argument("--target-transforms", default=",".join(DEFAULT_TARGET_TRANSFORMS))
    p.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    p.add_argument("--out", default="submission.csv")
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
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building excess-target feature panel")
    panel = build_features(prices, index_df)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df, val_df = splits["train_df"], splits["val_df"]
    print(f"   base train: {len(train_df):,} rows up to {splits['train_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {splits['val_start'].date()}")

    print(">> Training recent-window excess-target MLP")
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
    print(
        "   selected lookback/half_life/target/config/policy: "
        f"{best_lookback} ({best_start_marker}) / {best_half_life} / "
        f"{best_target_transform} / {best_config} / {best_policy}"
    )
    print(f"   selected recent train rows/dates: {best_train_rows:,} / {best_train_dates}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in val_metrics:
        print(
            "   validation mean 3d returns "
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
    top_k, method = _policy_choice(best_policy, pred_df, _index_state(index_df))
    weights = build_portfolio_custom(pred_df, top_k=top_k, method=method)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if args.json_out:
        payload = {
            "target": TARGET_COLUMN,
            "selected_lookback": best_lookback,
            "selected_train_window_start": best_start_marker,
            "selected_train_rows": best_train_rows,
            "selected_train_dates": best_train_dates,
            "selected_half_life": best_half_life,
            "selected_target_transform": best_target_transform,
            "selected_config": best_config,
            "selected_allocation_policy": best_policy,
            "prediction_top_k": int(top_k),
            "prediction_weight_method": method,
            "model_configs": MLP_CONFIGS,
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "half_lives": half_lives,
            "target_transforms": target_transforms,
            "policies": policies,
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
