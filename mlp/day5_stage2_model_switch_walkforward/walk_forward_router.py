"""
Walk-forward robustness evaluation for the recent-window/style-dynamic router.
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
    from .router_common import _choose_router_by_walk_forward
except ImportError:  # noqa: E402
    from router_common import _choose_router_by_walk_forward

from mlp.day5_stage2_recent_window.features import build_features as build_recent_features  # noqa: E402
from mlp.day5_stage2_style_dynamic.features import build_features as build_style_features  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    recent_panel = build_recent_features(prices, index_df)
    style_panel = build_style_features(prices, index_df)

    result = _choose_router_by_walk_forward(recent_panel, style_panel, index_df, as_of=args.as_of)

    print(">> Router walk-forward selection")
    print(f"   selected router policy: {result['router_policy']}")
    print(
        "   walk-forward mean excess/std/objective: "
        f"{result['validation']['mean_excess_return']*100:+.3f}% / "
        f"{result['validation']['std_excess_return']*100:.3f}% / "
        f"{result['validation']['objective']*100:+.3f}%"
    )
    print(
        f"   walk-forward positive excess rate: {result['validation']['positive_excess_rate']*100:.1f}%"
    )
    print(
        f"   walk-forward style share: {result['validation']['style_share']*100:.1f}%"
    )
    print(
        f"   negative window count: {int(result['validation']['negative_window_count'])}"
    )
    for row in result["window_results"]:
        test = row["test"]
        print(
            f"   window {row['window_id']}: "
            f"{row['split']['test_start']} to {row['split']['test_end']} | "
            f"excess {test['mean_excess_return']*100:+.3f}% | "
            f"style share {test['style_share']*100:.1f}%"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f">> Wrote walk-forward summary to {out_path}")


if __name__ == "__main__":
    main()
