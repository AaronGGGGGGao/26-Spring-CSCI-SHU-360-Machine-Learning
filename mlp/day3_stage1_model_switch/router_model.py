"""
Development entrypoint for the stage1 model-switch branch.
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

from baseline.paths import DATA_DIR  # noqa: E402
try:  # noqa: E402
    from .router_common import dev_submission_payload, fit_dev_models
except ImportError:  # noqa: E402
    from router_common import dev_submission_payload, fit_dev_models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Fitting tuned + style-dynamic branch models")
    dev_ctx = fit_dev_models(prices, index_df, as_of=args.as_of)
    bounds = dev_ctx["bounds"]
    print(f"   train end: {bounds['train_end'].date()}")
    print(f"   val: {bounds['val_start'].date()} to {bounds['val_end'].date()}")
    print(
        "   tuned branch: "
        f"{dev_ctx['tuned_selection']['config']} / "
        f"{dev_ctx['tuned_selection']['top_k']} / "
        f"{dev_ctx['tuned_selection']['weight_method']}"
    )
    print(
        "   style branch: "
        f"{dev_ctx['style_selection']['config']} / "
        f"{dev_ctx['style_selection']['half_life']} / "
        f"{dev_ctx['style_selection']['allocation_policy']}"
    )
    print(
        "   router policy: "
        f"{dev_ctx['router_selection']['router_policy']} "
        f"(val excess {dev_ctx['router_selection']['validation']['mean_excess_return']*100:+.3f}%)"
    )

    print(">> Predicting routed portfolio")
    out, meta = dev_submission_payload(dev_ctx, as_of=args.as_of)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"   selected branch on prediction date: {meta['selected_branch']}")
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if args.json_out:
        payload = {
            "prediction": meta,
            "tuned_selection": dev_ctx["tuned_selection"],
            "style_selection": dev_ctx["style_selection"],
            "router_selection": dev_ctx["router_selection"],
            "split": {
                "train_end": bounds["train_end"].date().isoformat(),
                "val_start": bounds["val_start"].date().isoformat(),
                "val_end": bounds["val_end"].date().isoformat(),
            },
        }
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f">> Wrote summary to {out_json}")


if __name__ == "__main__":
    main()

