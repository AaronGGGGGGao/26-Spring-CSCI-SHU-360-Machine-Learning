"""
Lightweight rolling comparison for the two stage1 finalist models.

This script intentionally avoids any broad hyperparameter search:
it reuses the already selected finalist configurations and only checks
whether they remain competitive across a small number of rolling windows.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402

VAL_DAYS = 10
TEST_DAYS = 10
XGB_FINAL = {
    "config_name": "shallow_reg",
    "top_k": 30,
    "weight_method": "score_sq",
    "half_life": 20,
}
MLP_FINAL = {
    "config_name": "narrow_deep",
    "top_k": 30,
    "weight_method": "softmax",
}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent)
    added = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        added = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.pop(0)
    return module


_xgb_features = _load_module("finalist_xgb_features", PROJECT_ROOT / "xgboost/day3_stage1/features.py")
_xgb_model = _load_module("finalist_xgb_model", PROJECT_ROOT / "xgboost/day3_stage1/xgboost_model.py")
_mlp_features = _load_module("finalist_mlp_features", PROJECT_ROOT / "mlp/day3_stage1/features.py")
_mlp_model = _load_module("finalist_mlp_model", PROJECT_ROOT / "mlp/day3_stage1/mlp_model.py")

build_xgb_features = _xgb_features.build_features
xgb_training_frame = _xgb_features.training_frame
XGB_HORIZON = _xgb_features.FORWARD_HORIZON
XGB_CONFIGS = {cfg["name"]: cfg for cfg in _xgb_model.MODEL_CONFIGS}
train_xgb_model = _xgb_model.train_model
evaluate_xgb = _xgb_model.evaluate_model

build_mlp_features = _mlp_features.build_features
mlp_training_frame = _mlp_features.training_frame
MLP_CONFIGS = {cfg["name"]: cfg for cfg in _mlp_model.MLP_CONFIGS}
make_mlp_model = _mlp_model.make_model
evaluate_mlp = _mlp_model.evaluate_model


def build_train_cutoff(panel: pd.DataFrame, as_of: pd.Timestamp, horizon: int):
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - horizon)
    return pd.Timestamp(trading_dates[cutoff_idx])


def split_from_pool(pool: pd.DataFrame):
    all_dates = np.sort(pool["date"].unique())
    need = TEST_DAYS + VAL_DAYS + 2 * EMBARGO_DAYS + 20
    if len(all_dates) < need:
        raise RuntimeError(f"Not enough dates for rolling split: need at least {need}, got {len(all_dates)}.")
    test_start = pd.Timestamp(all_dates[-TEST_DAYS])
    val_end_idx = -(TEST_DAYS + EMBARGO_DAYS + 1)
    val_end = pd.Timestamp(all_dates[val_end_idx])
    val_start = pd.Timestamp(all_dates[val_end_idx - VAL_DAYS + 1])
    train_end_idx = -(TEST_DAYS + EMBARGO_DAYS + VAL_DAYS + EMBARGO_DAYS + 1)
    train_end = pd.Timestamp(all_dates[train_end_idx])
    return {
        "train_df": pool[pool["date"] <= train_end].copy(),
        "val_df": pool[(pool["date"] >= val_start) & (pool["date"] <= val_end)].copy(),
        "test_df": pool[pool["date"] >= test_start].copy(),
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": pd.Timestamp(pool[pool["date"] >= test_start]["date"].max()),
    }


def choose_as_of_dates(panel_dates: np.ndarray, n_windows: int):
    trading_dates = np.sort(panel_dates)
    anchors = []
    idx = len(trading_dates) - 1
    while idx >= 0 and len(anchors) < n_windows:
        anchors.append(pd.Timestamp(trading_dates[idx]))
        idx -= TEST_DAYS
    return anchors


def run_window(index_df: pd.DataFrame, as_of: pd.Timestamp, xgb_panel: pd.DataFrame, mlp_panel: pd.DataFrame):
    rows = []

    xgb_cutoff = build_train_cutoff(xgb_panel, as_of, XGB_HORIZON)
    xgb_pool = xgb_training_frame(xgb_panel, max_date=xgb_cutoff)
    xgb_split = split_from_pool(xgb_pool)
    xgb_model = train_xgb_model(
        xgb_split["train_df"],
        xgb_split["val_df"],
        XGB_CONFIGS[XGB_FINAL["config_name"]],
        XGB_FINAL["half_life"],
    )
    xgb_val = evaluate_xgb(xgb_model, xgb_split["val_df"], index_df, top_k=XGB_FINAL["top_k"], weight_method=XGB_FINAL["weight_method"])
    xgb_test = evaluate_xgb(xgb_model, xgb_split["test_df"], index_df, top_k=XGB_FINAL["top_k"], weight_method=XGB_FINAL["weight_method"])
    rows.append(
        {
            "model": "xgboost_day3_stage1",
            "as_of": as_of.date().isoformat(),
            "test_start": xgb_split["test_start"].date().isoformat(),
            "test_end": xgb_split["test_end"].date().isoformat(),
            "selected_config": XGB_FINAL["config_name"],
            "selected_top_k": XGB_FINAL["top_k"],
            "selected_weight_method": XGB_FINAL["weight_method"],
            "selected_extra": XGB_FINAL["half_life"],
            "val_mean_excess_return": xgb_val.get("mean_excess_return"),
            "val_positive_excess_rate": xgb_val.get("positive_excess_rate"),
            **{f"test_{k}": v for k, v in xgb_test.items()},
        }
    )

    mlp_cutoff = build_train_cutoff(mlp_panel, as_of, XGB_HORIZON)
    mlp_pool = mlp_training_frame(mlp_panel, max_date=mlp_cutoff)
    mlp_split = split_from_pool(mlp_pool)
    mlp_model = make_mlp_model(MLP_CONFIGS[MLP_FINAL["config_name"]])
    mlp_model.fit(mlp_split["train_df"][_mlp_features.FEATURE_COLUMNS], mlp_split["train_df"][_mlp_features.TARGET_COLUMN])
    mlp_val = evaluate_mlp(mlp_model, mlp_split["val_df"], index_df, top_k=MLP_FINAL["top_k"], weight_method=MLP_FINAL["weight_method"])
    mlp_test = evaluate_mlp(mlp_model, mlp_split["test_df"], index_df, top_k=MLP_FINAL["top_k"], weight_method=MLP_FINAL["weight_method"])
    rows.append(
        {
            "model": "mlp_day3_stage1",
            "as_of": as_of.date().isoformat(),
            "test_start": mlp_split["test_start"].date().isoformat(),
            "test_end": mlp_split["test_end"].date().isoformat(),
            "selected_config": MLP_FINAL["config_name"],
            "selected_top_k": MLP_FINAL["top_k"],
            "selected_weight_method": MLP_FINAL["weight_method"],
            "selected_extra": None,
            "val_mean_excess_return": mlp_val.get("mean_excess_return"),
            "val_positive_excess_rate": mlp_val.get("positive_excess_rate"),
            **{f"test_{k}": v for k, v in mlp_test.items()},
        }
    )

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)

    print(">> Building cached feature panels")
    xgb_panel = build_xgb_features(prices, index_df)
    mlp_panel = build_mlp_features(prices, index_df)
    as_of_dates = choose_as_of_dates(xgb_panel["date"].unique(), args.windows)

    rows = []
    for as_of in as_of_dates:
        print(f">> Finalist rolling window anchored at {as_of.date()}")
        rows.extend(run_window(index_df, as_of, xgb_panel, mlp_panel))

    result = pd.DataFrame(rows)
    summary = (
        result.groupby("model")[["test_mean_excess_return", "test_positive_excess_rate", "test_rank_ic"]]
        .mean()
        .sort_values("test_mean_excess_return", ascending=False)
        .reset_index()
    )
    print(summary.to_string(index=False))

    if args.json_out:
        out = {
            "windows": args.windows,
            "xgb_final": XGB_FINAL,
            "mlp_final": MLP_FINAL,
            "records": rows,
            "summary": summary.to_dict(orient="records"),
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f">> Wrote finalist rolling summary to {out_path}")


if __name__ == "__main__":
    main()
