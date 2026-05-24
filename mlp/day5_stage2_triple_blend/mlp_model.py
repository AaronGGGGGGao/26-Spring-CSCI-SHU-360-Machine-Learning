"""
Stage 2 5-day triple score blend.

This branch blends three independently trained alpha models:
recent-window MLP, style-dynamic MLP, and recent-window MLP with explicit
5-day features. It tests whether the strong canonical 5-day feature alpha can
improve the current recent/style blend without relying on it as a standalone
model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
from mlp.day5_stage2_recent_window.features import (  # noqa: E402
    FEATURE_COLUMNS as RECENT_FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features as build_recent_features,
    prediction_frame as recent_prediction_frame,
    training_frame as recent_training_frame,
)
from mlp.day5_stage2_recent_window.mlp_model import (  # noqa: E402
    MLP_CONFIGS as RECENT_CONFIGS,
    build_portfolio_custom,
    parse_int_list,
    parse_lookbacks,
    parse_str_list,
    select_with_lookback,
)
from mlp.day5_stage2_style_dynamic.features import (  # noqa: E402
    FEATURE_COLUMNS as STYLE_FEATURE_COLUMNS,
    build_features as build_style_features,
    prediction_frame as style_prediction_frame,
    training_frame as style_training_frame,
)
from mlp.day5_stage2_style_dynamic.mlp_model import (  # noqa: E402
    MLP_CONFIGS as STYLE_CONFIGS,
    parse_int_list as parse_half_lives,
    select_model_and_policy,
)
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

VAL_DAYS = 10
DEFAULT_ALPHAS = [0.25, 0.40, 0.50, 0.60, 0.75]
DEFAULT_TOP_KS = [30, 35, 40]
DEFAULT_WEIGHT_METHODS = [
    "softmax_1.5",
    "softmax_1.8",
    "softmax_2.0",
    "softmax_risk_1.5_0.50",
]
DEFAULT_STYLE_POLICIES = ["static_softmax_2.0", "trend_dynamic", "breadth_dynamic", "defensive_dynamic"]
DEFAULT_TRIPLE_WEIGHTS = [
    (0.60, 0.40, 0.00),
    (0.50, 0.35, 0.15),
    (0.45, 0.35, 0.20),
    (0.40, 0.40, 0.20),
    (0.35, 0.40, 0.25),
    (0.30, 0.40, 0.30),
]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_weight_triples(text: str) -> list[tuple[float, float, float]]:
    triples: list[tuple[float, float, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals = [float(x.strip()) for x in item.split(":")]
        if len(vals) != 3:
            raise ValueError(f"blend weight triple must be recent:style:feature5d, got {item!r}")
        total = sum(vals)
        if total <= 0:
            raise ValueError(f"blend weight triple must have positive sum, got {item!r}")
        triples.append((vals[0] / total, vals[1] / total, vals[2] / total))
    return triples


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    def per_date(s: pd.Series) -> pd.Series:
        std = s.std()
        if pd.isna(std) or std <= 1e-12:
            return s * 0.0
        return (s - s.mean()) / std

    return df.groupby("date")[col].transform(per_date)


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


def _split_bounds(panel: pd.DataFrame, val_days: int, test_days: int | None = None, as_of: str | None = None) -> dict:
    as_of_ts = pd.Timestamp(as_of) if as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    pool = recent_training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(pool["date"].unique())

    if test_days is None:
        if len(all_dates) < val_days + EMBARGO_DAYS + 20:
            raise RuntimeError("Not enough dates to train; download more history.")
        val_start = pd.Timestamp(all_dates[-val_days])
        train_end = pd.Timestamp(all_dates[-(val_days + EMBARGO_DAYS + 1)])
        return {"train_cutoff": train_cutoff, "train_end": train_end, "val_start": val_start, "val_end": pd.Timestamp(all_dates[-1])}

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


def _branch_frames(
    recent_panel: pd.DataFrame,
    style_panel: pd.DataFrame,
    feature5d_panel: pd.DataFrame,
    bounds: dict,
    include_test: bool = False,
) -> dict:
    recent_pool = recent_training_frame(recent_panel, max_date=bounds["train_cutoff"])
    style_pool = style_training_frame(style_panel, max_date=bounds["train_cutoff"])
    feature5d_pool = feature5d_training_frame(feature5d_panel, max_date=bounds["train_cutoff"])

    out = {
        "recent_train": recent_pool[recent_pool["date"] <= bounds["train_end"]].copy(),
        "recent_val": recent_pool[(recent_pool["date"] >= bounds["val_start"]) & (recent_pool["date"] <= bounds["val_end"])].copy(),
        "style_train": style_pool[style_pool["date"] <= bounds["train_end"]].copy(),
        "style_val": style_pool[(style_pool["date"] >= bounds["val_start"]) & (style_pool["date"] <= bounds["val_end"])].copy(),
        "feature5d_train": feature5d_pool[feature5d_pool["date"] <= bounds["train_end"]].copy(),
        "feature5d_val": feature5d_pool[(feature5d_pool["date"] >= bounds["val_start"]) & (feature5d_pool["date"] <= bounds["val_end"])].copy(),
    }
    if include_test:
        out["recent_test"] = recent_pool[(recent_pool["date"] >= bounds["test_start"]) & (recent_pool["date"] <= bounds["test_end"])].copy()
        out["style_test"] = style_pool[(style_pool["date"] >= bounds["test_start"]) & (style_pool["date"] <= bounds["test_end"])].copy()
        out["feature5d_test"] = feature5d_pool[(feature5d_pool["date"] >= bounds["test_start"]) & (feature5d_pool["date"] <= bounds["test_end"])].copy()
    return out


def train_branch_models(frames: dict, index_df: pd.DataFrame, top_ks: list[int], methods: list[str], lookbacks: list[str], half_lives: list[int], style_policies: list[str]) -> dict:
    recent = select_with_lookback(
        frames["recent_train"],
        frames["recent_val"],
        index_df,
        RECENT_CONFIGS,
        top_ks,
        methods,
        lookbacks,
    )
    style = select_model_and_policy(
        frames["style_train"],
        frames["style_val"],
        index_df,
        STYLE_CONFIGS,
        half_lives,
        style_policies,
    )
    feature5d = select_feature5d_with_lookback(
        frames["feature5d_train"],
        frames["feature5d_val"],
        index_df,
        FEATURE5D_CONFIGS,
        top_ks,
        methods,
        lookbacks,
    )
    return {"recent": recent, "style": style, "feature5d": feature5d}


def _score_frame(
    recent_model,
    style_model,
    feature5d_model,
    recent_frame: pd.DataFrame,
    style_frame: pd.DataFrame,
    feature5d_frame: pd.DataFrame,
) -> pd.DataFrame:
    recent_scored = recent_frame[["date", "stock_code", TARGET_COLUMN, "idio_vol_20d"]].copy()
    recent_scored["recent_score"] = recent_model.predict(recent_frame[RECENT_FEATURE_COLUMNS])

    style_scored = style_frame[["date", "stock_code"]].copy()
    style_scored["style_score"] = style_model.predict(style_frame[STYLE_FEATURE_COLUMNS])

    feature5d_scored = feature5d_frame[["date", "stock_code"]].copy()
    feature5d_scored["feature5d_score"] = feature5d_model.predict(feature5d_frame[FEATURE5D_COLUMNS])

    scored = recent_scored.merge(style_scored, on=["date", "stock_code"], how="inner")
    scored = scored.merge(feature5d_scored, on=["date", "stock_code"], how="inner")
    scored["recent_z"] = _zscore_by_date(scored, "recent_score")
    scored["style_z"] = _zscore_by_date(scored, "style_score")
    scored["feature5d_z"] = _zscore_by_date(scored, "feature5d_score")
    return scored


def _blend_score(scored: pd.DataFrame, weights: tuple[float, float, float]) -> pd.Series:
    wr, ws, wf = weights
    return wr * scored["recent_z"] + ws * scored["style_z"] + wf * scored["feature5d_z"]


def _period_excess_return(scored: pd.DataFrame, index_df: pd.DataFrame, weights: tuple[float, float, float], top_k: int, method: str):
    frame = scored.copy()
    frame["score"] = _blend_score(frame, weights)

    index_panel = index_df.sort_values("date").copy()
    index_panel["date"] = pd.to_datetime(index_panel["date"])
    index_panel["bench_target"] = index_panel["close"].shift(-FORWARD_HORIZON) / index_panel["close"] - 1.0
    bench_fwd = index_panel.set_index("date")["bench_target"]

    rows = []
    for d, daily in frame.groupby("date"):
        bench_return = bench_fwd.get(pd.Timestamp(d))
        if pd.isna(bench_return):
            continue
        weights = build_portfolio_custom(daily, top_k=top_k, method=method)
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


def evaluate_blend(scored: pd.DataFrame, index_df: pd.DataFrame, weights: tuple[float, float, float], top_k: int, method: str) -> dict[str, float]:
    score = _blend_score(scored, weights)
    ic = rank_ic(scored[TARGET_COLUMN].to_numpy(), score.to_numpy(), scored["date"].to_numpy())
    _, bt = _period_excess_return(scored, index_df, weights=weights, top_k=top_k, method=method)
    out = {"rank_ic": float(ic)}
    if bt is not None:
        out.update(bt)
    return out


def select_blend(scored_val: pd.DataFrame, index_df: pd.DataFrame, weight_triples: list[tuple[float, float, float]], top_ks: list[int], methods: list[str]):
    leaderboard = []
    best_key = None
    best = None
    for weights in weight_triples:
        for top_k in top_ks:
            for method in methods:
                metrics = evaluate_blend(scored_val, index_df, weights=weights, top_k=top_k, method=method)
                row = {
                    "blend_recent_weight": float(weights[0]),
                    "blend_style_weight": float(weights[1]),
                    "blend_feature5d_weight": float(weights[2]),
                    "top_k": int(top_k),
                    "weight_method": method,
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
                    best = (weights, int(top_k), method, metrics)
    if best is None:
        raise RuntimeError("No blend candidates were evaluated.")
    return (*best, leaderboard)


def build_prediction(
    recent_model,
    style_model,
    feature5d_model,
    recent_panel: pd.DataFrame,
    style_panel: pd.DataFrame,
    feature5d_panel: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: str | None,
    weights: tuple[float, float, float],
    top_k: int,
    method: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    recent_pred = recent_prediction_frame(recent_panel, as_of=as_of).copy()
    style_pred = style_prediction_frame(style_panel, as_of=as_of).copy()
    feature5d_pred = feature5d_prediction_frame(feature5d_panel, as_of=as_of).copy()
    scored = _score_frame(recent_model, style_model, feature5d_model, recent_pred, style_pred, feature5d_pred)
    scored["score"] = _blend_score(scored, weights)
    pred_date = pd.Timestamp(scored["date"].iloc[0])
    weights = build_portfolio_custom(scored, top_k=top_k, method=method)
    return pd.DataFrame({"stock_code": weights.index, "weight": weights.values}), pred_date


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35,40")
    p.add_argument("--weight-methods", default="softmax_1.5,softmax_1.8,softmax_2.0,softmax_risk_1.5_0.50")
    p.add_argument("--lookbacks", default="126,189,252")
    p.add_argument("--half-lives", default="10,20,40")
    p.add_argument("--style-policies", default="static_softmax_2.0,trend_dynamic,breadth_dynamic,defensive_dynamic")
    p.add_argument(
        "--blend-weights",
        default="0.60:0.40:0.00,0.50:0.35:0.15,0.45:0.35:0.20,0.40:0.40:0.20,0.35:0.40:0.25,0.30:0.40:0.30",
        help="Comma-separated recent:style:feature5d blend triples.",
    )
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)
    lookbacks = parse_lookbacks(args.lookbacks)
    half_lives = parse_half_lives(args.half_lives)
    style_policies = parse_str_list(args.style_policies)
    blend_weights = parse_weight_triples(args.blend_weights)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building recent-window, style-dynamic, and 5d-feature panels")
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)
    feature5d_panel = build_feature5d_features(prices, index_df)
    bounds = _split_bounds(recent_panel, val_days=VAL_DAYS, as_of=args.as_of)
    frames = _branch_frames(recent_panel, style_panel, feature5d_panel, bounds)
    print(f"   train<= {bounds['train_end'].date()} | val {bounds['val_start'].date()} to {bounds['val_end'].date()}")

    print(">> Training branch models")
    branch = train_branch_models(frames, index_df, top_ks, methods, lookbacks, half_lives, style_policies)
    recent_model = branch["recent"][0]
    style_model = branch["style"][0]
    feature5d_model = branch["feature5d"][0]

    print(">> Selecting score blend")
    scored_val = _score_frame(
        recent_model,
        style_model,
        feature5d_model,
        frames["recent_val"],
        frames["style_val"],
        frames["feature5d_val"],
    )
    best_weights, best_top_k, best_method, val_metrics, blend_leaderboard = select_blend(
        scored_val, index_df, blend_weights, top_ks, methods
    )
    print(
        "   selected blend recent/style/feature5d, top_k, weight: "
        f"{best_weights[0]:.2f}/{best_weights[1]:.2f}/{best_weights[2]:.2f} / {best_top_k} / {best_method}"
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
        print(f"   validation positive excess rate: {val_metrics['positive_excess_rate']*100:.1f}% over {int(val_metrics['n_dates'])} dates")

    print(">> Predicting blended portfolio")
    out, pred_date = build_prediction(
        recent_model,
        style_model,
        feature5d_model,
        recent_panel,
        style_panel,
        feature5d_panel,
        index_df,
        args.as_of,
        best_weights,
        best_top_k,
        best_method,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {pred_date.date()}, wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if args.json_out:
        payload = {
            "selected_blend_recent_weight": best_weights[0],
            "selected_blend_style_weight": best_weights[1],
            "selected_blend_feature5d_weight": best_weights[2],
            "selected_top_k": best_top_k,
            "selected_weight_method": best_method,
            "branch_recent": {
                "selected_lookback": branch["recent"][1],
                "selected_train_window_start": branch["recent"][2],
                "selected_config": branch["recent"][5],
                "selected_top_k": branch["recent"][6],
                "selected_weight_method": branch["recent"][7],
                "validation": branch["recent"][8],
            },
            "branch_style": {
                "selected_config": branch["style"][1],
                "selected_half_life": branch["style"][2],
                "selected_allocation_policy": branch["style"][3],
                "validation": branch["style"][4],
            },
            "branch_feature5d": {
                "selected_lookback": branch["feature5d"][1],
                "selected_train_window_start": branch["feature5d"][2],
                "selected_config": branch["feature5d"][5],
                "selected_top_k": branch["feature5d"][6],
                "selected_weight_method": branch["feature5d"][7],
                "validation": branch["feature5d"][8],
            },
            "validation": val_metrics,
            "blend_leaderboard": blend_leaderboard,
            "prediction_date": pred_date.date().isoformat(),
        }
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f">> Wrote summary to {out_json}")


if __name__ == "__main__":
    main()
