"""
Robust self-test for the stage1 model-switch branch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import EMBARGO_DAYS  # noqa: E402
from baseline.paths import DATA_DIR  # noqa: E402
try:  # noqa: E402
    from .router_common import fit_self_test_models
except ImportError:  # noqa: E402
    from router_common import fit_self_test_models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    result = fit_self_test_models(prices, index_df, val_days=15, test_days=20, as_of=args.as_of)
    bounds = result["bounds"]

    print(">> Robust self-test split")
    print(f"   train: {result['split_rows']['train_rows']:,} rows up to {bounds['train_end'].date()}")
    print(f"   val:   {result['split_rows']['val_rows']:,} rows from {bounds['val_start'].date()} to {bounds['val_end'].date()}")
    print(f"   test:  {result['split_rows']['test_rows']:,} rows from {bounds['test_start'].date()} to {bounds['test_end'].date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days")
    print(">> Training model-switch branch (robust split)")
    print(
        "   recent branch: "
        f"{result['recent_selection']['lookback']} / "
        f"{result['recent_selection']['config']} / "
        f"{result['recent_selection']['top_k']} / "
        f"{result['recent_selection']['weight_method']}"
    )
    print(
        "   style branch: "
        f"{result['style_selection']['config']} / "
        f"{result['style_selection']['half_life']} / "
        f"{result['style_selection']['allocation_policy']}"
    )
    print(f"   selected router policy: {result['router_selection']['router_policy']}")
    print(
        "   router walk-forward mean excess/std/objective: "
        f"{result['router_selection']['validation']['mean_excess_return']*100:+.3f}% / "
        f"{result['router_selection']['validation']['std_excess_return']*100:.3f}% / "
        f"{result['router_selection']['validation']['objective']*100:+.3f}%"
    )
    print(
        "   router walk-forward positive/style/negative-window-count: "
        f"{result['router_selection']['validation']['positive_excess_rate']*100:.1f}% / "
        f"{result['router_selection']['validation']['style_share']*100:.1f}% / "
        f"{int(result['router_selection']['validation']['negative_window_count'])}"
    )
    print(
        "   test mean 5d returns "
        f"(portfolio/benchmark/excess): "
        f"{result['test']['mean_portfolio_return']*100:+.3f}% / "
        f"{result['test']['mean_benchmark_return']*100:+.3f}% / "
        f"{result['test']['mean_excess_return']*100:+.3f}%"
    )
    print(f"   test positive excess rate: {result['test']['positive_excess_rate']*100:.1f}% over {int(result['test']['n_dates'])} dates")

    if args.json_out:
        payload = {
            "split": {
                "train_end": bounds["train_end"].date().isoformat(),
                "val_start": bounds["val_start"].date().isoformat(),
                "val_end": bounds["val_end"].date().isoformat(),
                "test_start": bounds["test_start"].date().isoformat(),
                "test_end": bounds["test_end"].date().isoformat(),
                "train_rows": result["split_rows"]["train_rows"],
                "val_rows": result["split_rows"]["val_rows"],
                "test_rows": result["split_rows"]["test_rows"],
                "forward_horizon": int(3),
                "embargo_days": int(EMBARGO_DAYS),
                "val_days": int(15),
                "test_days": int(20),
            },
            "recent_selection": result["recent_selection"],
            "style_selection": result["style_selection"],
            "router_selection": result["router_selection"],
            "test": result["test"],
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f">> Wrote robust self-test summary to {out_path}")


if __name__ == "__main__":
    main()
