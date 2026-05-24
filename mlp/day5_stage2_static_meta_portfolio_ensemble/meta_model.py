"""
Stage 2 5-day static meta portfolio ensemble.

This branch avoids daily regime switching. It builds four complete candidate
portfolios, selects one fixed convex meta-weight vector on validation, and then
uses that same vector on the held-out test or final prediction date.
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

from baseline.baseline_xgboost import MAX_WEIGHT, MIN_STOCKS  # noqa: E402
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
    combine_portfolios,
    evaluate_portfolios,
    feature5d_prediction_frame,
    parse_float_list,
    parse_half_lives,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    recent_prediction_frame,
    select_portfolio_ensemble,
    style_prediction_frame,
    train_components,
)
from mlp.day5_stage2_recent_window.mlp_model import build_portfolio_custom  # noqa: E402
from mlp.day5_stage2_triple_blend.mlp_model import (  # noqa: E402
    _blend_score as triple_blend_score,
    _score_frame as triple_score_frame,
    parse_weight_triples,
    select_blend as select_triple_blend,
)


VAL_DAYS = 10
DEFAULT_TRIPLE_WEIGHTS = (
    "0.60:0.40:0.00,"
    "0.50:0.35:0.15,"
    "0.45:0.35:0.20,"
    "0.40:0.40:0.20,"
    "0.35:0.40:0.25,"
    "0.30:0.40:0.30"
)
DEFAULT_META_WEIGHTS = (
    "1.00:0.00:0.00:0.00,"
    "0.00:1.00:0.00:0.00,"
    "0.00:0.00:1.00:0.00,"
    "0.00:0.00:0.00:1.00,"
    "0.80:0.20:0.00:0.00,"
    "0.70:0.30:0.00:0.00,"
    "0.60:0.40:0.00:0.00,"
    "0.50:0.50:0.00:0.00,"
    "0.70:0.20:0.10:0.00,"
    "0.70:0.20:0.00:0.10,"
    "0.60:0.20:0.20:0.00,"
    "0.60:0.20:0.00:0.20,"
    "0.50:0.30:0.10:0.10,"
    "0.40:0.30:0.20:0.10"
)


def parse_weight_quads(text: str) -> list[tuple[float, float, float, float]]:
    quads = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals = [float(x.strip()) for x in item.split(":")]
        if len(vals) != 4:
            raise ValueError(f"expected pair:triple:style:feature weights, got {item!r}")
        total = sum(vals)
        if total <= 0:
            raise ValueError(f"weight quad must have positive sum, got {item!r}")
        quads.append(tuple(x / total for x in vals))
    return quads


def combine_meta_weight_dicts(
    pair_portfolio: dict[str, float],
    triple_portfolio: dict[str, float],
    style_portfolio: dict[str, float],
    feature_portfolio: dict[str, float],
    weights: tuple[float, float, float, float],
) -> dict[str, float]:
    wp, wt, ws, wf = weights
    names = sorted(set(pair_portfolio) | set(triple_portfolio) | set(style_portfolio) | set(feature_portfolio))
    combined = pd.Series(
        {
            name: wp * pair_portfolio.get(name, 0.0)
            + wt * triple_portfolio.get(name, 0.0)
            + ws * style_portfolio.get(name, 0.0)
            + wf * feature_portfolio.get(name, 0.0)
            for name in names
        },
        dtype=float,
    )
    combined = combined[combined > 1e-12]
    if len(combined) < MIN_STOCKS:
        raise RuntimeError(f"static meta portfolio produced only {len(combined)} names")
    if combined.max() > MAX_WEIGHT + 1e-8:
        raise RuntimeError(f"static meta portfolio violates {MAX_WEIGHT:.0%} cap")
    combined = combined / combined.sum()
    return _weights_to_dict(combined)


def combine_meta_portfolios(
    pair_portfolios: dict[pd.Timestamp, dict[str, float]],
    triple_portfolios: dict[pd.Timestamp, dict[str, float]],
    style_portfolios: dict[pd.Timestamp, dict[str, float]],
    feature_portfolios: dict[pd.Timestamp, dict[str, float]],
    weights: tuple[float, float, float, float],
) -> dict[pd.Timestamp, dict[str, float]]:
    dates = sorted(set(pair_portfolios) & set(triple_portfolios) & set(style_portfolios) & set(feature_portfolios))
    return {
        date: combine_meta_weight_dicts(
            pair_portfolios[date],
            triple_portfolios[date],
            style_portfolios[date],
            feature_portfolios[date],
            weights,
        )
        for date in dates
    }


def build_triple_portfolios(
    recent_model,
    style_model,
    feature5d_model,
    recent_frame: pd.DataFrame,
    style_frame: pd.DataFrame,
    feature5d_frame: pd.DataFrame,
    weights: tuple[float, float, float],
    top_k: int,
    method: str,
) -> dict[pd.Timestamp, dict[str, float]]:
    scored = triple_score_frame(recent_model, style_model, feature5d_model, recent_frame, style_frame, feature5d_frame)
    scored["score"] = triple_blend_score(scored, weights)
    portfolios = {}
    for d, daily in scored.groupby("date"):
        w = build_portfolio_custom(daily, top_k=top_k, method=method)
        portfolios[pd.Timestamp(d)] = _weights_to_dict(w)
    return portfolios


def selection_objective(metrics: dict[str, float], vol_penalty: float, downside_penalty: float, positive_bonus: float) -> float:
    return float(
        metrics["mean_excess_return"]
        - vol_penalty * metrics["excess_std"]
        + positive_bonus * metrics["positive_excess_rate"]
        + downside_penalty * min(0.0, metrics["min_excess_return"])
    )


def select_meta_weights(
    pair_portfolios: dict[pd.Timestamp, dict[str, float]],
    triple_portfolios: dict[pd.Timestamp, dict[str, float]],
    style_portfolios: dict[pd.Timestamp, dict[str, float]],
    feature_portfolios: dict[pd.Timestamp, dict[str, float]],
    realized_frame: pd.DataFrame,
    index_df: pd.DataFrame,
    meta_weights: list[tuple[float, float, float, float]],
    vol_penalty: float,
    downside_penalty: float,
    positive_bonus: float,
) -> tuple[tuple[float, float, float, float], dict[str, float], list[dict]]:
    leaderboard = []
    best_key = None
    best = None
    for weights in meta_weights:
        combined = combine_meta_portfolios(
            pair_portfolios,
            triple_portfolios,
            style_portfolios,
            feature_portfolios,
            weights,
        )
        _, metrics = evaluate_portfolios(combined, realized_frame, index_df)
        objective = selection_objective(metrics, vol_penalty, downside_penalty, positive_bonus)
        row = {
            "pair_weight": float(weights[0]),
            "triple_weight": float(weights[1]),
            "style_weight": float(weights[2]),
            "feature_weight": float(weights[3]),
            "objective": objective,
            **metrics,
        }
        leaderboard.append(row)
        key = (objective, metrics["mean_excess_return"], metrics["positive_excess_rate"], -metrics["excess_std"])
        if best_key is None or key > best_key:
            best_key = key
            best = (weights, {**metrics, "objective": objective})
    if best is None:
        raise RuntimeError("No meta weights were evaluated.")
    return (*best, leaderboard)


def fit_static_meta(
    frames: dict,
    feature_frames: dict,
    index_df: pd.DataFrame,
    top_ks: list[int],
    methods: list[str],
    lookbacks: list[str],
    half_lives: list[int],
    style_policies: list[str],
    score_blend_weights: list[float],
    pair_weights: list[float],
    triple_weights: list[tuple[float, float, float]],
    meta_weights: list[tuple[float, float, float, float]],
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
    branch = components["branch"]
    recent_model = branch["recent"][0]
    style_model = branch["style"][0]
    feature_model = feature_cfg["model"]

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
        feature_model,
        feature_frames["feature_val"],
        feature_cfg["selected_top_k"],
        feature_cfg["selected_weight_method"],
    )
    pair_weight, pair_val_metrics, pair_leaderboard = select_portfolio_ensemble(
        style_val_ports,
        feature_val_ports,
        frames["recent_val"],
        index_df,
        pair_weights,
        vol_penalty,
        downside_penalty,
        positive_bonus,
    )
    pair_val_ports = combine_portfolios(style_val_ports, feature_val_ports, pair_weight)

    scored_triple_val = triple_score_frame(
        recent_model,
        style_model,
        feature_model,
        frames["recent_val"],
        frames["style_val"],
        feature_frames["feature_val"],
    )
    triple_w, triple_top_k, triple_method, triple_val_metrics, triple_leaderboard = select_triple_blend(
        scored_triple_val,
        index_df,
        triple_weights,
        top_ks,
        methods,
    )
    triple_val_ports = build_triple_portfolios(
        recent_model,
        style_model,
        feature_model,
        frames["recent_val"],
        frames["style_val"],
        feature_frames["feature_val"],
        triple_w,
        triple_top_k,
        triple_method,
    )
    meta_w, meta_val_metrics, meta_leaderboard = select_meta_weights(
        pair_val_ports,
        triple_val_ports,
        style_val_ports,
        feature_val_ports,
        frames["recent_val"],
        index_df,
        meta_weights,
        vol_penalty,
        downside_penalty,
        positive_bonus,
    )
    return {
        "components": components,
        "selected_pair_weight": pair_weight,
        "pair_validation": pair_val_metrics,
        "pair_leaderboard": pair_leaderboard,
        "selected_triple_weights": triple_w,
        "selected_triple_top_k": triple_top_k,
        "selected_triple_weight_method": triple_method,
        "triple_validation": triple_val_metrics,
        "triple_leaderboard": triple_leaderboard,
        "selected_meta_weights": meta_w,
        "validation": meta_val_metrics,
        "meta_leaderboard": meta_leaderboard,
    }


def build_child_portfolios_for_frame(
    bundle: dict,
    recent_frame: pd.DataFrame,
    style_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> tuple[
    dict[pd.Timestamp, dict[str, float]],
    dict[pd.Timestamp, dict[str, float]],
    dict[pd.Timestamp, dict[str, float]],
    dict[pd.Timestamp, dict[str, float]],
]:
    components = bundle["components"]
    style_cfg = components["style_recent"]
    feature_cfg = components["feature5d"]
    branch = components["branch"]
    feature_model = feature_cfg["model"]

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
        feature_model,
        feature_frame,
        feature_cfg["selected_top_k"],
        feature_cfg["selected_weight_method"],
    )
    pair_ports = combine_portfolios(style_ports, feature_ports, bundle["selected_pair_weight"])
    triple_ports = build_triple_portfolios(
        branch["recent"][0],
        branch["style"][0],
        feature_model,
        recent_frame,
        style_frame,
        feature_frame,
        bundle["selected_triple_weights"],
        bundle["selected_triple_top_k"],
        bundle["selected_triple_weight_method"],
    )
    return pair_ports, triple_ports, style_ports, feature_ports


def evaluate_static_meta(
    bundle: dict,
    recent_frame: pd.DataFrame,
    style_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    index_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    child_ports = build_child_portfolios_for_frame(bundle, recent_frame, style_frame, feature_frame)
    combined = combine_meta_portfolios(*child_ports, bundle["selected_meta_weights"])
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
    child_ports = build_child_portfolios_for_frame(bundle, recent_pred, style_pred, feature_pred)
    pred_date = sorted(child_ports[0])[0]
    combined = combine_meta_portfolios(*child_ports, bundle["selected_meta_weights"])
    weights = pd.Series(combined[pred_date], dtype=float).sort_values(ascending=False)
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values}), pred_date


def bundle_summary(bundle: dict) -> dict:
    mw = bundle["selected_meta_weights"]
    tw = bundle["selected_triple_weights"]
    return {
        "selected_meta_pair_weight": mw[0],
        "selected_meta_triple_weight": mw[1],
        "selected_meta_style_weight": mw[2],
        "selected_meta_feature_weight": mw[3],
        "selected_pair_weight": bundle["selected_pair_weight"],
        "selected_triple_recent_weight": tw[0],
        "selected_triple_style_weight": tw[1],
        "selected_triple_feature5d_weight": tw[2],
        "selected_triple_top_k": bundle["selected_triple_top_k"],
        "selected_triple_weight_method": bundle["selected_triple_weight_method"],
        "validation": bundle["validation"],
        "pair_validation": bundle["pair_validation"],
        "triple_validation": bundle["triple_validation"],
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
    p.add_argument("--pair-weights", default="0.50,0.60,0.70,0.80,0.90,1.00")
    p.add_argument("--triple-weights", default=DEFAULT_TRIPLE_WEIGHTS)
    p.add_argument("--meta-weights", default=DEFAULT_META_WEIGHTS)
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
    pair_weights = parse_float_list(args.pair_weights)
    triple_weights = parse_weight_triples(args.triple_weights)
    meta_weights = parse_weight_quads(args.meta_weights)

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

    print(">> Training children and selecting static meta weights")
    bundle = fit_static_meta(
        frames,
        feature_frames,
        index_df,
        top_ks,
        methods,
        lookbacks,
        half_lives,
        style_policies,
        score_blend_weights,
        pair_weights,
        triple_weights,
        meta_weights,
        args.vol_penalty,
        args.downside_penalty,
        args.positive_bonus,
    )
    mw = bundle["selected_meta_weights"]
    print(f"   selected meta weights pair/triple/style/feature: {mw[0]:.2f} / {mw[1]:.2f} / {mw[2]:.2f} / {mw[3]:.2f}")
    print(
        "   validation mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{bundle['validation']['mean_portfolio_return']*100:+.3f}% / "
        f"{bundle['validation']['mean_benchmark_return']*100:+.3f}% / "
        f"{bundle['validation']['mean_excess_return']*100:+.3f}%"
    )

    print(">> Predicting static-meta portfolio")
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
            "meta_leaderboard": bundle["meta_leaderboard"],
            "prediction_date": pred_date.date().isoformat(),
        }
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f">> Wrote summary to {out_json}")


if __name__ == "__main__":
    main()
