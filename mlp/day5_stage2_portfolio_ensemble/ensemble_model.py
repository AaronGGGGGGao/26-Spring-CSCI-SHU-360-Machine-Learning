"""
Stage 2 5-day portfolio-level ensemble.

This branch keeps the strongest existing 5-day alphas separate until the final
portfolio construction step. Each child model first builds a valid long-only
portfolio, then the final model averages portfolio weights and selects the
averaging coefficient with a stability-aware validation objective.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS, MAX_WEIGHT, MIN_STOCKS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
from mlp.day5_stage2_recent_window_5d_features.features import (  # noqa: E402
    FEATURE_COLUMNS as FEATURE5D_COLUMNS,
    build_features as build_feature5d_features,
    prediction_frame as feature5d_prediction_frame,
    training_frame as feature5d_training_frame,
)
from mlp.day5_stage2_recent_window_5d_features.mlp_model import (  # noqa: E402
    MLP_CONFIGS as FEATURE5D_CONFIGS,
    select_with_lookback as select_feature5d_with_lookback,
)
from mlp.day5_stage2_recent_window.mlp_model import build_portfolio_custom  # noqa: E402
from mlp.day5_stage2_style_recent_blend.mlp_model import (  # noqa: E402
    FORWARD_HORIZON,
    RECENT_FEATURE_COLUMNS,
    RECENT_CONFIGS,
    STYLE_CONFIGS,
    STYLE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    _branch_frames,
    _score_frame,
    _split_bounds,
    build_recent_features,
    build_style_features,
    parse_float_list,
    parse_half_lives,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    recent_prediction_frame,
    select_blend,
    style_prediction_frame,
    train_branch_models,
)


VAL_DAYS = 10
DEFAULT_PORTFOLIO_WEIGHTS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def _feature5d_frames(panel: pd.DataFrame, bounds: dict, include_test: bool = False) -> dict:
    pool = feature5d_training_frame(panel, max_date=bounds["train_cutoff"])
    out = {
        "feature_train": pool[pool["date"] <= bounds["train_end"]].copy(),
        "feature_val": pool[(pool["date"] >= bounds["val_start"]) & (pool["date"] <= bounds["val_end"])].copy(),
    }
    if include_test:
        out["feature_test"] = pool[(pool["date"] >= bounds["test_start"]) & (pool["date"] <= bounds["test_end"])].copy()
    return out


def _benchmark_forward(index_df: pd.DataFrame) -> pd.Series:
    index_panel = index_df.sort_values("date").copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel["bench_target"] = index_panel["close"].shift(-FORWARD_HORIZON) / index_panel["close"] - 1.0
    return index_panel.set_index("date")["bench_target"]


def _weights_to_dict(weights: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in weights.items() if float(v) > 0.0}


def _portfolio_weights_by_date(scored: pd.DataFrame, top_k: int, method: str) -> dict[pd.Timestamp, dict[str, float]]:
    portfolios = {}
    for d, daily in scored.groupby("date"):
        weights = build_portfolio_custom(daily, top_k=top_k, method=method)
        portfolios[pd.Timestamp(d)] = _weights_to_dict(weights)
    return portfolios


def build_style_recent_portfolios(
    recent_model,
    style_model,
    recent_frame: pd.DataFrame,
    style_frame: pd.DataFrame,
    blend_recent_weight: float,
    top_k: int,
    method: str,
) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    scored = _score_frame(recent_model, style_model, recent_frame, style_frame)
    scored["score"] = blend_recent_weight * scored["recent_z"] + (1.0 - blend_recent_weight) * scored["style_z"]
    return _portfolio_weights_by_date(scored, top_k=top_k, method=method), scored


def build_feature5d_portfolios(
    model,
    frame: pd.DataFrame,
    top_k: int,
    method: str,
) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    scored = frame[["date", "stock_code", TARGET_COLUMN, "idio_vol_20d"]].copy()
    scored["score"] = model.predict(frame[FEATURE5D_COLUMNS])
    return _portfolio_weights_by_date(scored, top_k=top_k, method=method), scored


def combine_weight_dicts(
    style_portfolio: dict[str, float],
    feature_portfolio: dict[str, float],
    style_weight: float,
) -> dict[str, float]:
    names = sorted(set(style_portfolio) | set(feature_portfolio))
    combined = pd.Series(
        {
            name: style_weight * style_portfolio.get(name, 0.0)
            + (1.0 - style_weight) * feature_portfolio.get(name, 0.0)
            for name in names
        },
        dtype=float,
    )
    combined = combined[combined > 1e-12]
    if len(combined) < MIN_STOCKS:
        raise RuntimeError(f"portfolio ensemble produced only {len(combined)} names")
    if combined.max() > MAX_WEIGHT + 1e-9:
        raise RuntimeError(f"portfolio ensemble violates {MAX_WEIGHT:.0%} cap")
    combined = combined / combined.sum()
    return _weights_to_dict(combined)


def combine_portfolios(
    style_portfolios: dict[pd.Timestamp, dict[str, float]],
    feature_portfolios: dict[pd.Timestamp, dict[str, float]],
    style_weight: float,
) -> dict[pd.Timestamp, dict[str, float]]:
    dates = sorted(set(style_portfolios) & set(feature_portfolios))
    return {
        date: combine_weight_dicts(style_portfolios[date], feature_portfolios[date], style_weight)
        for date in dates
    }


def evaluate_portfolios(
    portfolios: dict[pd.Timestamp, dict[str, float]],
    realized_frame: pd.DataFrame,
    index_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    bench_fwd = _benchmark_forward(index_df)
    rows = []
    for d in sorted(portfolios):
        bench_return = bench_fwd.get(pd.Timestamp(d))
        if pd.isna(bench_return):
            continue
        daily = realized_frame[realized_frame["date"] == pd.Timestamp(d)].set_index("stock_code")
        weights = pd.Series(portfolios[d], dtype=float)
        realized = daily[TARGET_COLUMN].reindex(weights.index)
        if realized.isna().any():
            continue
        portfolio_return = float((weights * realized).sum())
        rows.append(
            {
                "date": pd.Timestamp(d),
                "portfolio_return": portfolio_return,
                "benchmark_return": float(bench_return),
                "excess_return": portfolio_return - float(bench_return),
                "n_names": int((weights > 0).sum()),
                "max_weight": float(weights.max()),
            }
        )

    series = pd.DataFrame(rows).sort_values("date")
    if series.empty:
        raise RuntimeError("No portfolio dates could be evaluated.")
    excess = series["excess_return"]
    metrics = {
        "n_dates": float(len(series)),
        "mean_portfolio_return": float(series["portfolio_return"].mean()),
        "mean_benchmark_return": float(series["benchmark_return"].mean()),
        "mean_excess_return": float(excess.mean()),
        "excess_std": float(excess.std(ddof=0)),
        "min_excess_return": float(excess.min()),
        "positive_excess_rate": float((excess > 0).mean()),
        "mean_n_names": float(series["n_names"].mean()),
        "max_weight": float(series["max_weight"].max()),
    }
    return series, metrics


def selection_objective(
    metrics: dict[str, float],
    vol_penalty: float,
    downside_penalty: float,
    positive_bonus: float,
) -> float:
    downside = min(0.0, metrics["min_excess_return"])
    return float(
        metrics["mean_excess_return"]
        - vol_penalty * metrics["excess_std"]
        + positive_bonus * metrics["positive_excess_rate"]
        + downside_penalty * downside
    )


def select_portfolio_ensemble(
    style_portfolios: dict[pd.Timestamp, dict[str, float]],
    feature_portfolios: dict[pd.Timestamp, dict[str, float]],
    realized_frame: pd.DataFrame,
    index_df: pd.DataFrame,
    portfolio_weights: list[float],
    vol_penalty: float,
    downside_penalty: float,
    positive_bonus: float,
) -> tuple[float, dict[str, float], list[dict]]:
    leaderboard = []
    best_key = None
    best = None
    for style_weight in portfolio_weights:
        combined = combine_portfolios(style_portfolios, feature_portfolios, style_weight)
        _, metrics = evaluate_portfolios(combined, realized_frame, index_df)
        objective = selection_objective(metrics, vol_penalty, downside_penalty, positive_bonus)
        row = {"style_recent_portfolio_weight": float(style_weight), "objective": objective, **metrics}
        leaderboard.append(row)
        key = (
            objective,
            metrics["mean_excess_return"],
            metrics["positive_excess_rate"],
            -metrics["excess_std"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (float(style_weight), {**metrics, "objective": objective})
    if best is None:
        raise RuntimeError("No portfolio ensemble candidates were evaluated.")
    return (*best, leaderboard)


def train_components(
    frames: dict,
    feature_frames: dict,
    index_df: pd.DataFrame,
    top_ks: list[int],
    methods: list[str],
    lookbacks: list[str],
    half_lives: list[int],
    style_policies: list[str],
    score_blend_weights: list[float],
) -> dict:
    branch = train_branch_models(frames, index_df, top_ks, methods, lookbacks, half_lives, style_policies)
    recent_model = branch["recent"][0]
    style_model = branch["style"][0]
    scored_val = _score_frame(recent_model, style_model, frames["recent_val"], frames["style_val"])
    alpha, style_top_k, style_method, style_val_metrics, style_leaderboard = select_blend(
        scored_val, index_df, score_blend_weights, top_ks, methods
    )

    feature = select_feature5d_with_lookback(
        feature_frames["feature_train"],
        feature_frames["feature_val"],
        index_df,
        FEATURE5D_CONFIGS,
        top_ks,
        methods,
        lookbacks,
    )

    return {
        "branch": branch,
        "style_recent": {
            "recent_model": recent_model,
            "style_model": style_model,
            "blend_recent_weight": alpha,
            "top_k": style_top_k,
            "weight_method": style_method,
            "validation": style_val_metrics,
            "leaderboard": style_leaderboard,
        },
        "feature5d": {
            "model": feature[0],
            "selected_lookback": feature[1],
            "selected_train_window_start": feature[2],
            "selected_train_rows": feature[3],
            "selected_train_dates": feature[4],
            "selected_config": feature[5],
            "selected_top_k": feature[6],
            "selected_weight_method": feature[7],
            "validation": feature[8],
            "leaderboard": feature[9],
        },
    }


def build_prediction(
    components: dict,
    recent_panel: pd.DataFrame,
    style_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    as_of: str | None,
    style_recent_portfolio_weight: float,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    style_cfg = components["style_recent"]
    feature_cfg = components["feature5d"]
    recent_pred = recent_prediction_frame(recent_panel, as_of=as_of).copy()
    style_pred = style_prediction_frame(style_panel, as_of=as_of).copy()
    feature_pred = feature5d_prediction_frame(feature_panel, as_of=as_of).copy()

    style_ports, style_scored = build_style_recent_portfolios(
        style_cfg["recent_model"],
        style_cfg["style_model"],
        recent_pred,
        style_pred,
        style_cfg["blend_recent_weight"],
        style_cfg["top_k"],
        style_cfg["weight_method"],
    )
    feature_ports, _ = build_feature5d_portfolios(
        feature_cfg["model"],
        feature_pred,
        feature_cfg["selected_top_k"],
        feature_cfg["selected_weight_method"],
    )
    pred_date = pd.Timestamp(style_scored["date"].iloc[0])
    combined = combine_portfolios(style_ports, feature_ports, style_recent_portfolio_weight)
    weights = pd.Series(combined[pred_date], dtype=float).sort_values(ascending=False)
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
        return {str(k): _json_safe(v) for k, v in obj.items() if k not in {"model", "recent_model", "style_model"}}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35")
    p.add_argument("--weight-methods", default="softmax_1.8,softmax_2.0")
    p.add_argument("--lookbacks", default="126,189")
    p.add_argument("--half-lives", default="10,40")
    p.add_argument("--style-policies", default="static_softmax_2.0,trend_dynamic,breadth_dynamic,defensive_dynamic")
    p.add_argument("--score-blend-weights", default="0.25,0.40,0.50,0.60,0.75")
    p.add_argument("--portfolio-weights", default="0.50,0.60,0.70,0.80,0.90,1.00")
    p.add_argument("--vol-penalty", type=float, default=0.25)
    p.add_argument("--downside-penalty", type=float, default=0.25)
    p.add_argument("--positive-bonus", type=float, default=0.002)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    half_lives = parse_half_lives(args.half_lives)
    style_policies = parse_str_list(args.style_policies)
    score_blend_weights = parse_float_list(args.score_blend_weights)
    portfolio_weights = parse_float_list(args.portfolio_weights)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building feature panels")
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    feature_panel = build_feature5d_features(prices, index_df)
    bounds = _split_bounds(recent_panel, val_days=VAL_DAYS, as_of=args.as_of)
    frames = _branch_frames(recent_panel, style_panel, bounds)
    feature_frames = _feature5d_frames(feature_panel, bounds)
    print(f"   train<= {bounds['train_end'].date()} | val {bounds['val_start'].date()} to {bounds['val_end'].date()}")
    print(f"   horizon/embargo: {FORWARD_HORIZON} / {EMBARGO_DAYS} trading days")

    print(">> Training child models")
    components = train_components(
        frames,
        feature_frames,
        index_df,
        top_ks,
        methods,
        lookbacks,
        half_lives,
        style_policies,
        score_blend_weights,
    )

    print(">> Selecting portfolio-level ensemble")
    style_cfg = components["style_recent"]
    feature_cfg = components["feature5d"]
    style_val_ports, _ = build_style_recent_portfolios(
        style_cfg["recent_model"],
        style_cfg["style_model"],
        frames["recent_val"],
        frames["style_val"],
        style_cfg["blend_recent_weight"],
        style_cfg["top_k"],
        style_cfg["weight_method"],
    )
    feature_val_ports, _ = build_feature5d_portfolios(
        feature_cfg["model"],
        feature_frames["feature_val"],
        feature_cfg["selected_top_k"],
        feature_cfg["selected_weight_method"],
    )
    ensemble_weight, ensemble_val_metrics, ensemble_leaderboard = select_portfolio_ensemble(
        style_val_ports,
        feature_val_ports,
        frames["recent_val"],
        index_df,
        portfolio_weights,
        args.vol_penalty,
        args.downside_penalty,
        args.positive_bonus,
    )
    print(f"   selected style_recent_portfolio_weight: {ensemble_weight:.2f}")
    print(
        "   validation mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{ensemble_val_metrics['mean_portfolio_return']*100:+.3f}% / "
        f"{ensemble_val_metrics['mean_benchmark_return']*100:+.3f}% / "
        f"{ensemble_val_metrics['mean_excess_return']*100:+.3f}%"
    )
    print(
        f"   validation objective/std/positive/min: {ensemble_val_metrics['objective']*100:+.3f}% / "
        f"{ensemble_val_metrics['excess_std']*100:.3f}% / "
        f"{ensemble_val_metrics['positive_excess_rate']*100:.1f}% / "
        f"{ensemble_val_metrics['min_excess_return']*100:+.3f}%"
    )

    print(">> Predicting portfolio-level ensemble")
    out, pred_date = build_prediction(components, recent_panel, style_panel, feature_panel, args.as_of, ensemble_weight)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {pred_date.date()}, wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if args.json_out:
        payload = {
            "horizon": FORWARD_HORIZON,
            "target_column": TARGET_COLUMN,
            "selection_objective": {
                "vol_penalty": args.vol_penalty,
                "downside_penalty": args.downside_penalty,
                "positive_bonus": args.positive_bonus,
            },
            "selected_style_recent_portfolio_weight": ensemble_weight,
            "style_recent": {
                "blend_recent_weight": style_cfg["blend_recent_weight"],
                "top_k": style_cfg["top_k"],
                "weight_method": style_cfg["weight_method"],
                "validation": style_cfg["validation"],
            },
            "feature5d": {
                "selected_lookback": feature_cfg["selected_lookback"],
                "selected_train_window_start": feature_cfg["selected_train_window_start"],
                "selected_train_rows": feature_cfg["selected_train_rows"],
                "selected_train_dates": feature_cfg["selected_train_dates"],
                "selected_config": feature_cfg["selected_config"],
                "selected_top_k": feature_cfg["selected_top_k"],
                "selected_weight_method": feature_cfg["selected_weight_method"],
                "validation": feature_cfg["validation"],
            },
            "branch_recent": {
                "selected_lookback": components["branch"]["recent"][1],
                "selected_train_window_start": components["branch"]["recent"][2],
                "selected_config": components["branch"]["recent"][5],
                "selected_top_k": components["branch"]["recent"][6],
                "selected_weight_method": components["branch"]["recent"][7],
                "validation": components["branch"]["recent"][8],
            },
            "branch_style": {
                "selected_config": components["branch"]["style"][1],
                "selected_half_life": components["branch"]["style"][2],
                "selected_allocation_policy": components["branch"]["style"][3],
                "validation": components["branch"]["style"][4],
            },
            "validation": ensemble_val_metrics,
            "ensemble_leaderboard": ensemble_leaderboard,
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookbacks": lookbacks,
            "half_lives": half_lives,
            "style_policies": style_policies,
            "score_blend_weights": score_blend_weights,
            "portfolio_weights": portfolio_weights,
            "prediction_date": pred_date.date().isoformat(),
        }
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f">> Wrote summary to {out_json}")


if __name__ == "__main__":
    main()
