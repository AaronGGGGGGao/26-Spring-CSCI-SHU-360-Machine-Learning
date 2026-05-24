"""
Stage1-oriented 3-day LSTM sequence model.

This experiment keeps the same public-data 3-day feature panel, but uses a
small LSTM over the last 20 trading observations of each stock.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
LOOKBACK = 20
DEFAULT_TOP_KS = [30, 35]
DEFAULT_WEIGHT_METHODS = ["softmax_2.0", "score_sq"]
LSTM_CONFIGS = [
    {
        "name": "lstm_small",
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 10,
        "batch_size": 512,
    },
    {
        "name": "lstm_medium",
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "lr": 8e-4,
        "weight_decay": 2e-4,
        "epochs": 12,
        "batch_size": 512,
    },
]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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
    chosen = daily.sort_values("score", ascending=False).head(top_k).copy().set_index("stock_code")
    if method == "softmax_2.0":
        w = _softmax_weights(chosen["score"], 2.0)
    elif method == "score_sq":
        z = chosen["score"] - chosen["score"].min() + 1e-8
        raw = np.square(z)
        w = raw / raw.sum()
    else:
        raise ValueError(f"unknown weight method: {method}")
    return _cap_and_redistribute(w.astype(float))


def period_excess_return(frame: pd.DataFrame, pred: np.ndarray, index_df: pd.DataFrame, top_k: int, weight_method: str):
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
        weights = build_portfolio_custom(daily, top_k=top_k, method=weight_method)
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


@dataclass
class SequenceData:
    x: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    stock_codes: np.ndarray
    eval_frame: pd.DataFrame


def _base_ready_frame(panel: pd.DataFrame, require_target: bool) -> pd.DataFrame:
    cols = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    extra = ["date", "stock_code"]
    keep = panel.dropna(subset=cols).copy()
    return keep.sort_values(["stock_code", "date"]).reset_index(drop=True)


def build_sequence_dataset(df: pd.DataFrame, lookback: int, target_dates: set[pd.Timestamp] | None = None, require_target: bool = True) -> SequenceData:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    dates: list[pd.Timestamp] = []
    codes: list[str] = []
    rows: list[dict] = []

    for stock_code, g in df.groupby("stock_code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        feats = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        g_dates = pd.to_datetime(g["date"]).to_list()
        targets = g[TARGET_COLUMN].to_numpy(dtype=np.float32) if require_target else None
        for i in range(lookback - 1, len(g)):
            d = pd.Timestamp(g_dates[i])
            if target_dates is not None and d not in target_dates:
                continue
            window = feats[i - lookback + 1 : i + 1]
            if window.shape[0] != lookback:
                continue
            xs.append(window)
            ys.append(float(targets[i]) if require_target else math.nan)
            dates.append(d)
            codes.append(str(stock_code))
            rows.append({"date": d, "stock_code": str(stock_code), TARGET_COLUMN: float(targets[i]) if require_target else math.nan})

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    dates_arr = np.asarray(dates, dtype="datetime64[ns]")
    codes_arr = np.asarray(codes, dtype=object)
    eval_frame = pd.DataFrame(rows)
    return SequenceData(x=x, y=y, dates=dates_arr, stock_codes=codes_arr, eval_frame=eval_frame)


def build_prediction_sequences(panel: pd.DataFrame, as_of: pd.Timestamp, lookback: int) -> SequenceData:
    df = _base_ready_frame(panel, require_target=False)
    return build_sequence_dataset(df, lookback=lookback, target_dates={pd.Timestamp(as_of)}, require_target=False)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        last = h[-1]
        return self.head(last).squeeze(-1)


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
    std = flat.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(train_seq: SequenceData, val_seq: SequenceData, config: dict):
    mean, std = fit_scaler(train_seq.x)
    x_train = apply_scaler(train_seq.x, mean, std)
    x_val = apply_scaler(val_seq.x, mean, std)

    model = LSTMRegressor(
        input_size=x_train.shape[-1],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.SmoothL1Loss()

    train_loader = make_loader(x_train, train_seq.y, batch_size=config["batch_size"], shuffle=True)
    val_loader = make_loader(x_val, val_seq.y, batch_size=config["batch_size"], shuffle=False)

    best_state = None
    best_val = float("inf")
    patience = 3
    bad = 0

    for _ in range(config["epochs"]):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                losses.append(float(loss_fn(pred, yb).item()))
        val_loss = float(np.mean(losses))
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "mean": mean, "std": std}


def predict_bundle(bundle: dict, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model: LSTMRegressor = bundle["model"]
    mean, std = bundle["mean"], bundle["std"]
    x_scaled = apply_scaler(x, mean, std)
    loader = DataLoader(torch.from_numpy(x_scaled), batch_size=batch_size, shuffle=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for xb in loader:
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


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


def evaluate_model(bundle: dict, seq: SequenceData, index_df: pd.DataFrame, top_k: int, weight_method: str) -> dict[str, float]:
    pred = predict_bundle(bundle, seq.x)
    frame = seq.eval_frame.copy()
    ic = rank_ic(seq.y, pred, seq.dates)
    _, bt = period_excess_return(frame, pred, index_df, top_k=top_k, weight_method=weight_method)
    result = {"rank_ic": float(ic)}
    if bt is not None:
        result.update(bt)
    return result


def select_model_and_portfolio(train_seq, val_seq, index_df, configs, top_ks, methods):
    leaderboard = []
    best_key = None
    best = None
    for config in configs:
        bundle = train_model(train_seq, val_seq, config)
        for top_k in top_ks:
            for method in methods:
                metrics = evaluate_model(bundle, val_seq, index_df, top_k=top_k, weight_method=method)
                row = {
                    "config_name": config["name"],
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
                    best = (bundle, config["name"], int(top_k), method, metrics)
    assert best is not None
    return (*best, leaderboard)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-ks", default="30,35")
    p.add_argument("--weight-methods", default="softmax_2.0,score_sq")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    top_ks = parse_int_list(args.top_ks)
    methods = parse_str_list(args.weight_methods)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building sequence feature panel")
    panel = build_features(prices, index_df)
    splits = build_dev_split(panel, as_of=args.as_of)
    train_df, val_df = splits["train_df"], splits["val_df"]
    train_seq = build_sequence_dataset(train_df, lookback=LOOKBACK, require_target=True)
    val_dates = set(pd.to_datetime(val_df["date"]).unique())
    val_seq = build_sequence_dataset(pd.concat([train_df, val_df], ignore_index=True), lookback=LOOKBACK, target_dates=val_dates, require_target=True)
    print(f"   train rows: {len(train_df):,}, train seq: {len(train_seq.y):,}")
    print(f"   val rows:   {len(val_df):,}, val seq:   {len(val_seq.y):,}")
    print(f"   train end: {splits['train_end'].date()} | val start: {splits['val_start'].date()}")

    print(">> Training LSTM sequence model")
    bundle, best_config, best_top_k, best_method, val_metrics, leaderboard = select_model_and_portfolio(
        train_seq, val_seq, index_df, LSTM_CONFIGS, top_ks, methods
    )
    print(f"   selected config/top_k/weight: {best_config} / {best_top_k} / {best_method}")
    print(f"   validation rank IC: {val_metrics['rank_ic']:.4f}")
    if "mean_excess_return" in val_metrics:
        print(
            "   validation mean 5d returns "
            f"(portfolio/benchmark/excess): "
            f"{val_metrics['mean_portfolio_return']*100:+.3f}% / "
            f"{val_metrics['mean_benchmark_return']*100:+.3f}% / "
            f"{val_metrics['mean_excess_return']*100:+.3f}%"
        )

    print(">> Predicting portfolio")
    as_of = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    pred_seq = build_prediction_sequences(panel, as_of=as_of, lookback=LOOKBACK)
    pred = predict_bundle(bundle, pred_seq.x)
    pred_frame = pred_seq.eval_frame.copy()
    pred_frame["score"] = pred
    weights = build_portfolio_custom(pred_frame, top_k=best_top_k, method=best_method)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   as of {as_of.date()}, scoring {len(pred_frame)} stocks")
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if args.json_out:
        payload = {
            "selected_config": best_config,
            "selected_top_k": best_top_k,
            "selected_weight_method": best_method,
            "lstm_configs": LSTM_CONFIGS,
            "top_ks": top_ks,
            "weight_methods": methods,
            "lookback": LOOKBACK,
            "leaderboard": leaderboard,
            "validation": val_metrics,
            "prediction_date": as_of.date().isoformat(),
            "forward_horizon": FORWARD_HORIZON,
            "train_sequence_count": int(len(train_seq.y)),
            "validation_sequence_count": int(len(val_seq.y)),
        }
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f">> Wrote summary to {json_path}")


if __name__ == "__main__":
    main()
