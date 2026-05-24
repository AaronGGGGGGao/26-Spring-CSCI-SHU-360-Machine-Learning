"""
Stage 2 5-day regime-aware portfolio ensemble.

This branch extends the current best portfolio-level ensemble with a defensive
price-only child portfolio. The goal is not to maximize the strongest rebound
window, but to reduce weak-window drag while preserving the existing alpha
children and all portfolio constraints.
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
from mlp.day5_stage2_portfolio_ensemble.ensemble_model import (  # noqa: E402
    FORWARD_HORIZON,
    TARGET_COLUMN,
    _branch_frames,
    _feature5d_frames,
    _json_safe,
    _split_bounds,
    _weights_to_dict,
    build_feature5d_features,
    build_feature5d_portfolios,
    build_recent_features,
    build_style_features,
    build_style_recent_portfolios,
    evaluate_portfolios,
    feature5d_prediction_frame,
    parse_float_list,
    parse_half_lives,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    recent_prediction_frame,
    style_prediction_frame,
    train_components,
)
from mlp.day5_stage2_recent_window.mlp_model import build_portfolio_custom  # noqa: E402


VAL_DAYS = 10
DEFAULT_ENSEMBLE_WEIGHTS = [
    "1.00:0.00:0.00",
    "0.90:0.10:0.00",
    "0.80:0.20:0.00",
    "0.70:0.30:0.00",
    "0.60:0.40:0.00",
    "0.50:0.50:0.00",
    "0.90:0.00:0.10",
    "0.80:0.00:0.20",
    "0.80:0.10:0.10",
    "0.70:0.10:0.20",
    "0.60:0.20:0.20",
    "0.50:0.30:0.20",
]


def parse_weight_triples(text: str) -> list[tuple[float, float, float]]:
    triples = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [float(x.strip()) for x in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"expected weight triple style:feature:defensive, got {item!r}")
        total = sum(parts)
        if total <= 0:
            raise ValueError(f"weight triple must have positive sum: {item!r}")
        triples.append(tuple(x / total for x in parts))
    return triples


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    def per_date(s: pd.Series) -> pd.Series:
        std = s.std()
        if pd.isna(std) or std <= 1e-12:
            return s * 0.0
        return (s - s.mean()) / std

    return df.groupby("date")[col].transform(per_date)


def build_defensive_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame[["date", "stock_code", TARGET_COLUMN, "idio_vol_20d"]].copy()
    tmp = frame.copy()
    tmp["low_beta"] = 1.0 - tmp["beta_20d_rank"]
    tmp["low_idio_vol"] = 1.0 - tmp["idio_vol_20d_rank"]
    tmp["liquidity_rank"] = 0.5 * tmp["volume_z_10d_rank"] + 0.5 * tmp["turnover_z_10d_rank"]
    tmp["defensive_raw"] = (
        0.25 * tmp["excess_ret_5d_rank"]
        + 0.20 * tmp["relative_strength_5_10_rank"]
        + 0.20 * tmp["low_beta"]
        + 0.20 * tmp["low_idio_vol"]
        + 0.10 * tmp["liquidity_rank"]
        + 0.05 * tmp["close_pos_in_range_rank"]
    )
    scored["score"] = _zscore_by_date(tmp, "defensive_raw")
    return scored


def build_defensive_portfolios(
    frame: pd.DataFrame,
    top_k: int,
    method: str,
) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    scored = build_defensive_scores(frame)
    portfolios = {}
    for d, daily in scored.groupby("date"):
        weights = build_portfolio_custom(daily, top_k=top_k, method=method)
        portfolios[pd.Timestamp(d)] = _weights_to_dict(weights)
    return portfolios, scored


def combine_three_weight_dicts(
    style_portfolio: dict[str, float],
    feature_portfolio: dict[str, float],
    defensive_portfolio: dict[str, float],
    weights: tuple[float, float, float],
) -> dict[str, float]:
    sw, fw, dw = weights
    names = sorted(set(style_portfolio) | set(feature_portfolio) | set(defensive_portfolio))
    combined = pd.Series(
        {
            name: sw * style_portfolio.get(name, 0.0)
            + fw * feature_portfolio.get(name, 0.0)
            + dw * defensive_portfolio.get(name, 0.0)
            for name in names
        },
        dtype=float,
    )
    combined = combined[combined > 1e-12]
    if len(combined) < MIN_STOCKS:
        raise RuntimeError(f"regime ensemble produced only {len(combined)} names")
    if combined.max() > MAX_WEIGHT + 1e-8:
        raise RuntimeError(f"regime ensemble violates {MAX_WEIGHT:.0%} cap")
    combined = combined / combined.sum()
    return _weights_to_dict(combined)


def combine_three_portfolios(
    style_portfolios: dict[pd.Timestamp, dict[str, float]],
    feature_portfolios: dict[pd.Timestamp, dict[str, float]],
    defensive_portfolios: dict[pd.Timestamp, dict[str, float]],
    weights: tuple[float, float, float],
) -> dict[pd.Timestamp, dict[str, float]]:
    dates = sorted(set(style_portfolios) & set(feature_portfolios) & set(defensive_portfolios))
    return {
        date: combine_three_weight_dicts(
            style_portfolios[date],
            feature_portfolios[date],
            defensive_portfolios[date],
            weights,
        )
        for date in dates
    }


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


def select_regime_ensemble(
    style_portfolios: dict[pd.Timestamp, dict[str, float]],
    feature_portfolios: dict[pd.Timestamp, dict[str, float]],
    defensive_candidates: dict[tuple[int, str], dict[pd.Timestamp, dict[str, float]]],
    realized_frame: pd.DataFrame,
    index_df: pd.DataFrame,
    ensemble_weights: list[tuple[float, float, float]],
    vol_penalty: float,
    downside_penalty: float,
    positive_bonus: float,
) -> tuple[tuple[float, float, float], int, str, dict[str, float], list[dict]]:
    leaderboard = []
    best_key = None
    best = None
    for defensive_key, defensive_portfolios in defensive_candidates.items():
        defensive_top_k, defensive_method = defensive_key
        for weights in ensemble_weights:
            combined = combine_three_portfolios(style_portfolios, feature_portfolios, defensive_portfolios, weights)
            _, metrics = evaluate_portfolios(combined, realized_frame, index_df)
            objective = selection_objective(metrics, vol_penalty, downside_penalty, positive_bonus)
            row = {
                "style_recent_portfolio_weight": float(weights[0]),
                "feature5d_portfolio_weight": float(weights[1]),
                "defensive_portfolio_weight": float(weights[2]),
                "defensive_top_k": int(defensive_top_k),
                "defensive_weight_method": defensive_method,
                "objective": objective,
                **metrics,
            }
            leaderboard.append(row)
            key = (
                objective,
                metrics["positive_excess_rate"],
                metrics["mean_excess_return"],
                -metrics["excess_std"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (weights, int(defensive_top_k), defensive_method, {**metrics, "objective": objective})
    if best is None:
        raise RuntimeError("No regime ensemble candidates were evaluated.")
    return (*best, leaderboard)


def fit_regime_ensemble(
    frames: dict,
    feature_frames: dict,
    index_df: pd.DataFrame,
    top_ks: list[int],
    methods: list[str],
    lookbacks: list[str],
    half_lives: list[int],
    style_policies: list[str],
    score_blend_weights: list[float],
    defensive_top_ks: list[int],
    defensive_methods: list[str],
    ensemble_weights: list[tuple[float, float, float]],
    vol_penalty: float,
    downside_penalty: float,
    positive_bonus: float,
) -> dict:
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
    defensive_candidates = {}
    for top_k in defensive_top_ks:
        for method in defensive_methods:
            defensive_candidates[(top_k, method)] = build_defensive_portfolios(frames["recent_val"], top_k, method)[0]

    weights, defensive_top_k, defensive_method, val_metrics, leaderboard = select_regime_ensemble(
        style_val_ports,
        feature_val_ports,
        defensive_candidates,
        frames["recent_val"],
        index_df,
        ensemble_weights,
        vol_penalty,
        downside_penalty,
        positive_bonus,
    )
    return {
        "components": components,
        "selected_weights": weights,
        "selected_defensive_top_k": defensive_top_k,
        "selected_defensive_weight_method": defensive_method,
        "validation": val_metrics,
        "leaderboard": leaderboard,
    }


def evaluate_regime_bundle(
    bundle: dict,
    recent_frame: pd.DataFrame,
    style_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    index_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    components = bundle["components"]
    style_cfg = components["style_recent"]
    feature_cfg = components["feature5d"]
    style_ports, _ = build_style_recent_portfolios(
        style_cfg["recent_model"],
        style_cfg["style_model"],
        recent_frame,
        style_frame,
        style_cfg["blend_recent_weight"],
        style_cfg["top_k"],
        style_cfg["weight_method"],
    )
    feature_ports, _ = build_feature5d_portfolios(
        feature_cfg["model"],
        feature_frame,
        feature_cfg["selected_top_k"],
        feature_cfg["selected_weight_method"],
    )
    defensive_ports, _ = build_defensive_portfolios(
        recent_frame,
        bundle["selected_defensive_top_k"],
        bundle["selected_defensive_weight_method"],
    )
    combined = combine_three_portfolios(style_ports, feature_ports, defensive_ports, bundle["selected_weights"])
    return evaluate_portfolios(combined, recent_frame, index_df)


def build_prediction(
    bundle: dict,
    recent_panel: pd.DataFrame,
    style_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    as_of: str | None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    recent_pred = recent_prediction_frame(recent_panel, as_of=as_of).copy()
    style_pred = style_prediction_frame(style_panel, as_of=as_of).copy()
    feature_pred = feature5d_prediction_frame(feature_panel, as_of=as_of).copy()

    components = bundle["components"]
    style_cfg = components["style_recent"]
    feature_cfg = components["feature5d"]
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
    defensive_ports, _ = build_defensive_portfolios(
        recent_pred,
        bundle["selected_defensive_top_k"],
        bundle["selected_defensive_weight_method"],
    )
    pred_date = pd.Timestamp(style_scored["date"].iloc[0])
    combined = combine_three_portfolios(style_ports, feature_ports, defensive_ports, bundle["selected_weights"])
    weights = pd.Series(combined[pred_date], dtype=float).sort_values(ascending=False)
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values}), pred_date


def bundle_summary(bundle: dict) -> dict:
    components = bundle["components"]
    return {
        "selected_style_recent_portfolio_weight": bundle["selected_weights"][0],
        "selected_feature5d_portfolio_weight": bundle["selected_weights"][1],
        "selected_defensive_portfolio_weight": bundle["selected_weights"][2],
        "selected_defensive_top_k": bundle["selected_defensive_top_k"],
        "selected_defensive_weight_method": bundle["selected_defensive_weight_method"],
        "validation": bundle["validation"],
        "style_recent": {
            "blend_recent_weight": components["style_recent"]["blend_recent_weight"],
            "top_k": components["style_recent"]["top_k"],
            "weight_method": components["style_recent"]["weight_method"],
            "validation": components["style_recent"]["validation"],
        },
        "feature5d": {
            "selected_lookback": components["feature5d"]["selected_lookback"],
            "selected_config": components["feature5d"]["selected_config"],
            "selected_top_k": components["feature5d"]["selected_top_k"],
            "selected_weight_method": components["feature5d"]["selected_weight_method"],
            "validation": components["feature5d"]["validation"],
        },
    }


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
    p.add_argument("--defensive-top-ks", default="40,50")
    p.add_argument("--defensive-weight-methods", default="softmax_1.2,softmax_1.5")
    p.add_argument("--ensemble-weights", default=",".join(DEFAULT_ENSEMBLE_WEIGHTS))
    p.add_argument("--vol-penalty", type=float, default=0.50)
    p.add_argument("--downside-penalty", type=float, default=0.50)
    p.add_argument("--positive-bonus", type=float, default=0.004)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    half_lives = parse_half_lives(args.half_lives)
    style_policies = parse_str_list(args.style_policies)
    score_blend_weights = parse_float_list(args.score_blend_weights)
    defensive_top_ks = parse_int_list(args.defensive_top_ks)
    defensive_methods = parse_str_list(args.defensive_weight_methods)
    ensemble_weights = parse_weight_triples(args.ensemble_weights)

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

    print(">> Training alpha children and selecting defensive ensemble")
    bundle = fit_regime_ensemble(
        frames,
        feature_frames,
        index_df,
        top_ks,
        methods,
        lookbacks,
        half_lives,
        style_policies,
        score_blend_weights,
        defensive_top_ks,
        defensive_methods,
        ensemble_weights,
        args.vol_penalty,
        args.downside_penalty,
        args.positive_bonus,
    )
    sw, fw, dw = bundle["selected_weights"]
    print(f"   selected portfolio weights style/feature/defensive: {sw:.2f} / {fw:.2f} / {dw:.2f}")
    print(
        f"   defensive portfolio: top {bundle['selected_defensive_top_k']} / "
        f"{bundle['selected_defensive_weight_method']}"
    )
    print(
        "   validation mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{bundle['validation']['mean_portfolio_return']*100:+.3f}% / "
        f"{bundle['validation']['mean_benchmark_return']*100:+.3f}% / "
        f"{bundle['validation']['mean_excess_return']*100:+.3f}%"
    )

    print(">> Predicting regime-aware portfolio")
    out, pred_date = build_prediction(bundle, recent_panel, style_panel, feature_panel, args.as_of)
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
            **bundle_summary(bundle),
            "leaderboard": bundle["leaderboard"],
            "prediction_date": pred_date.date().isoformat(),
        }
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f">> Wrote summary to {out_json}")


if __name__ == "__main__":
    main()
