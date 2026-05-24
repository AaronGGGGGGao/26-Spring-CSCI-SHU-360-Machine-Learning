"""
Stage 2 5-day LightGBM LambdaRank model.

This branch directly optimizes daily cross-sectional ordering. Labels are
derived from same-day ranks of 5-day excess returns, while all model selection
and reported performance use held-out 5-day portfolio excess return.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
from mlp.day5_stage2_recent_window.mlp_model import (  # noqa: E402
    build_portfolio_custom,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    period_excess_return,
    rank_ic,
)
from mlp.day5_stage2_recent_window.features import FORWARD_HORIZON, TARGET_COLUMN  # noqa: E402
from mlp.day5_stage2_recent_window_5d_features.features import (  # noqa: E402
    EXTRA_FEATURE_COLUMNS,
    build_features as build_feature5d_features,
)
from mlp.day5_stage2_style_dynamic.features import (  # noqa: E402
    FEATURE_COLUMNS as STYLE_FEATURE_COLUMNS,
    build_features as build_style_features,
)


VAL_DAYS = 10
TARGET_EXCESS_COLUMN = "target_excess_5d"
FEATURE_COLUMNS = list(STYLE_FEATURE_COLUMNS) + [c for c in EXTRA_FEATURE_COLUMNS if c not in STYLE_FEATURE_COLUMNS]

LGBM_CONFIGS = [
    {
        "name": "ranker_leaf31_hl40",
        "num_leaves": 31,
        "learning_rate": 0.035,
        "n_estimators": 180,
        "min_child_samples": 35,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "recency_half_life": 40,
    },
    {
        "name": "ranker_leaf63_hl40",
        "num_leaves": 63,
        "learning_rate": 0.025,
        "n_estimators": 240,
        "min_child_samples": 45,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 1.5,
        "recency_half_life": 40,
    },
    {
        "name": "ranker_leaf31_hl80",
        "num_leaves": 31,
        "learning_rate": 0.030,
        "n_estimators": 220,
        "min_child_samples": 50,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 2.0,
        "recency_half_life": 80,
    },
]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _benchmark_forward(index_df: pd.DataFrame) -> pd.Series:
    index_panel = index_df.sort_values("date").copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel["bench_target"] = index_panel["close"].shift(-FORWARD_HORIZON) / index_panel["close"] - 1.0
    return index_panel.set_index("date")["bench_target"]


def build_features(prices: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    style_panel = build_style_features(prices, index_df).reset_index(drop=True)
    feature5d_panel = build_feature5d_features(prices, index_df).reset_index(drop=True)
    extra = feature5d_panel[["date", "stock_code", *EXTRA_FEATURE_COLUMNS]].copy()
    panel = style_panel.merge(extra, on=["date", "stock_code"], how="left")
    bench = _benchmark_forward(index_df).rename("bench_target_5d")
    panel = panel.merge(bench, left_on="date", right_index=True, how="left")
    panel[TARGET_EXCESS_COLUMN] = panel[TARGET_COLUMN] - panel["bench_target_5d"]
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, TARGET_EXCESS_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    return panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()


def _rank_labels(df: pd.DataFrame, bins: int) -> np.ndarray:
    pct = df.groupby("date")[TARGET_EXCESS_COLUMN].rank(method="average", pct=True)
    labels = np.floor(pct.to_numpy(dtype=float) * bins).astype(int)
    return np.clip(labels, 0, bins - 1)


def _group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("date", sort=False).size().astype(int).tolist()


def _recency_weights(df: pd.DataFrame, half_life: float) -> np.ndarray:
    unique_dates = np.sort(df["date"].unique())
    order = {pd.Timestamp(d): i for i, d in enumerate(unique_dates)}
    latest = len(unique_dates) - 1
    ages = df["date"].map(lambda d: latest - order[pd.Timestamp(d)]).to_numpy(dtype=float)
    return np.power(0.5, ages / float(half_life))


def _sort_for_ranker(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "stock_code"]).reset_index(drop=True)


def make_ranker(config: dict) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 3, 7, 15, 31, 63, 127, 255, 511],
        num_leaves=config["num_leaves"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        min_child_samples=config["min_child_samples"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_ranker(train_df: pd.DataFrame, config: dict, label_bins: int) -> lgb.LGBMRanker:
    train_sorted = _sort_for_ranker(train_df)
    y = _rank_labels(train_sorted, label_bins)
    group = _group_sizes(train_sorted)
    sample_weight = _recency_weights(train_sorted, config["recency_half_life"])
    model = make_ranker(config)
    model.fit(
        train_sorted[FEATURE_COLUMNS],
        y,
        group=group,
        sample_weight=sample_weight,
    )
    return model


def evaluate_model(
    model: lgb.LGBMRanker,
    frame: pd.DataFrame,
    index_df: pd.DataFrame,
    top_k: int,
    weight_method: str,
) -> dict[str, float]:
    pred = model.predict(frame[FEATURE_COLUMNS])
    ic = rank_ic(frame[TARGET_EXCESS_COLUMN].to_numpy(), pred, frame["date"].to_numpy())
    series, bt = period_excess_return(frame, pred, index_df, top_k=top_k, weight_method=weight_method)
    out = {"rank_ic": float(ic)}
    if bt is not None:
        excess = series["excess_return"]
        out.update(bt)
        out.update(
            {
                "excess_std": float(excess.std(ddof=0)),
                "min_excess_return": float(excess.min()),
            }
        )
    return out


def selection_objective(metrics: dict[str, float], vol_penalty: float, downside_penalty: float, positive_bonus: float) -> float:
    downside = min(0.0, metrics.get("min_excess_return", 0.0))
    return float(
        metrics.get("mean_excess_return", float("-inf"))
        - vol_penalty * metrics.get("excess_std", 0.0)
        + positive_bonus * metrics.get("positive_excess_rate", 0.0)
        + downside_penalty * downside
    )


def _restrict_recent_window(train_df: pd.DataFrame, lookback: str) -> tuple[pd.DataFrame, str]:
    if lookback == "full":
        return train_df.copy(), "full"
    lookback_days = int(lookback)
    train_dates = np.sort(train_df["date"].unique())
    if len(train_dates) < lookback_days:
        return train_df.copy(), "full_fallback"
    start_date = pd.Timestamp(train_dates[-lookback_days])
    return train_df[train_df["date"] >= start_date].copy(), start_date.date().isoformat()


def split_bounds(panel: pd.DataFrame, val_days: int, test_days: int | None = None, as_of: str | None = None) -> dict:
    as_of_ts = pd.Timestamp(as_of) if as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(pool["date"].unique())

    if test_days is None:
        if len(all_dates) < val_days + EMBARGO_DAYS + 20:
            raise RuntimeError("Not enough dates to train; download more history.")
        val_start = pd.Timestamp(all_dates[-val_days])
        train_end = pd.Timestamp(all_dates[-(val_days + EMBARGO_DAYS + 1)])
        return {
            "train_cutoff": train_cutoff,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": pd.Timestamp(all_dates[-1]),
        }

    need = test_days + val_days + 2 * EMBARGO_DAYS + 20
    if len(all_dates) < need:
        raise RuntimeError(f"Not enough dates for self-test split: need at least {need}, got {len(all_dates)}.")
    test_start = pd.Timestamp(all_dates[-test_days])
    test_end = pd.Timestamp(all_dates[-1])
    val_end_idx = -(test_days + EMBARGO_DAYS + 1)
    val_end = pd.Timestamp(all_dates[val_end_idx])
    val_start = pd.Timestamp(all_dates[val_end_idx - val_days + 1])
    train_end_idx = -(test_days + EMBARGO_DAYS + val_days + EMBARGO_DAYS + 1)
    train_end = pd.Timestamp(all_dates[train_end_idx])
    return {
        "train_cutoff": train_cutoff,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def frames_from_bounds(panel: pd.DataFrame, bounds: dict, include_test: bool = False) -> dict:
    pool = training_frame(panel, max_date=bounds["train_cutoff"])
    out = {
        "train": pool[pool["date"] <= bounds["train_end"]].copy(),
        "val": pool[(pool["date"] >= bounds["val_start"]) & (pool["date"] <= bounds["val_end"])].copy(),
    }
    if include_test:
        out["test"] = pool[(pool["date"] >= bounds["test_start"]) & (pool["date"] <= bounds["test_end"])].copy()
    return out


def select_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    index_df: pd.DataFrame,
    configs: list[dict],
    top_ks: list[int],
    methods: list[str],
    lookbacks: list[str],
    label_bins_list: list[int],
    vol_penalty: float,
    downside_penalty: float,
    positive_bonus: float,
) -> tuple[lgb.LGBMRanker, str, str, int, int, dict, int, int, str, dict, list[dict]]:
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
        for label_bins in label_bins_list:
            for config in configs:
                model = fit_ranker(recent_train, config, label_bins)
                for top_k in top_ks:
                    for method in methods:
                        metrics = evaluate_model(model, val_df, index_df, top_k=top_k, weight_method=method)
                        objective = selection_objective(metrics, vol_penalty, downside_penalty, positive_bonus)
                        row = {
                            "lookback": lookback,
                            "train_window_start": start_marker,
                            "train_rows": int(len(recent_train)),
                            "train_dates": int(unique_dates),
                            "label_bins": int(label_bins),
                            "config_name": config["name"],
                            "top_k": int(top_k),
                            "weight_method": method,
                            "objective": objective,
                            **metrics,
                        }
                        leaderboard.append(row)
                        key = (
                            objective,
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
                                config,
                                int(label_bins),
                                int(top_k),
                                method,
                                {**metrics, "objective": objective},
                            )
    if best is None:
        raise RuntimeError("All LightGBM ranker candidates were skipped.")
    return (*best, leaderboard)


def build_prediction(
    model: lgb.LGBMRanker,
    panel: pd.DataFrame,
    as_of: str | None,
    top_k: int,
    weight_method: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    pred_df = prediction_frame(panel, as_of=as_of).copy()
    if pred_df.empty:
        raise RuntimeError(f"No prediction rows available for as_of={as_of}.")
    pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS])
    pred_date = pd.Timestamp(pred_df["date"].iloc[0])
    weights = build_portfolio_custom(pred_df, top_k=top_k, method=weight_method)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    return out, pred_date


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.date().isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument("--weight-methods", default="softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.5_0.50")
    p.add_argument("--lookbacks", default="126,189")
    p.add_argument("--label-bins", default="5,10")
    p.add_argument("--vol-penalty", type=float, default=0.15)
    p.add_argument("--downside-penalty", type=float, default=0.20)
    p.add_argument("--positive-bonus", type=float, default=0.002)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    label_bins_list = parse_int_list(args.label_bins)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building 5-day LightGBM ranker feature panel")
    panel = build_features(prices, index_df)
    bounds = split_bounds(panel, val_days=VAL_DAYS, as_of=args.as_of)
    frames = frames_from_bounds(panel, bounds)
    print(f"   features: {len(FEATURE_COLUMNS)}")
    print(f"   train<= {bounds['train_end'].date()} | val {bounds['val_start'].date()} to {bounds['val_end'].date()}")
    print(f"   train cutoff for 5d label: {bounds['train_cutoff'].date()} | embargo: {EMBARGO_DAYS}")

    print(">> Training and selecting LightGBM ranker")
    (
        model,
        best_lookback,
        best_start_marker,
        best_train_rows,
        best_train_dates,
        best_config,
        best_label_bins,
        best_top_k,
        best_method,
        val_metrics,
        leaderboard,
    ) = select_ranker(
        frames["train"],
        frames["val"],
        index_df,
        LGBM_CONFIGS,
        top_ks,
        methods,
        lookbacks,
        label_bins_list,
        args.vol_penalty,
        args.downside_penalty,
        args.positive_bonus,
    )
    print(
        "   selected lookback/config/label_bins/top_k/weight: "
        f"{best_lookback} ({best_start_marker}) / {best_config['name']} / "
        f"{best_label_bins} / {best_top_k} / {best_method}"
    )
    print(f"   selected train rows/dates: {best_train_rows:,} / {best_train_dates}")
    print(
        "   validation mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
        f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
        f"{val_metrics['mean_excess_return']*100:+.3f}%"
    )
    print(
        f"   validation objective/rank_ic/positive/std/min: "
        f"{val_metrics['objective']*100:+.3f}% / {val_metrics['rank_ic']:.4f} / "
        f"{val_metrics['positive_excess_rate']*100:.1f}% / "
        f"{val_metrics['excess_std']*100:.3f}% / {val_metrics['min_excess_return']*100:+.3f}%"
    )

    print(">> Predicting LightGBM ranker portfolio")
    out, pred_date = build_prediction(model, panel, args.as_of, best_top_k, best_method)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {pred_date.date()}, wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if args.json_out:
        payload = {
            "model_family": "day5_stage2_lgbm_ranker",
            "horizon": FORWARD_HORIZON,
            "target_column": TARGET_COLUMN,
            "rank_label_column": TARGET_EXCESS_COLUMN,
            "feature_count": len(FEATURE_COLUMNS),
            "selected_lookback": best_lookback,
            "selected_train_window_start": best_start_marker,
            "selected_train_rows": best_train_rows,
            "selected_train_dates": best_train_dates,
            "selected_config": best_config,
            "selected_label_bins": best_label_bins,
            "selected_top_k": best_top_k,
            "selected_weight_method": best_method,
            "selection_objective": {
                "vol_penalty": args.vol_penalty,
                "downside_penalty": args.downside_penalty,
                "positive_bonus": args.positive_bonus,
            },
            "validation": val_metrics,
            "leaderboard": leaderboard,
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "label_bins": label_bins_list,
            "prediction_date": pred_date.date().isoformat(),
        }
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f">> Wrote summary to {out_json}")


if __name__ == "__main__":
    main()
