# Stage 2 5-Day Recent-Window MLP

## Purpose

This branch starts Stage 2 with a clean 5-trading-day model. It is based on the
Stage 1 winner, `mlp/day3_stage1_recent_window`, but every target, benchmark
return, split cutoff, printed metric, and JSON field uses a 5-day horizon.

The implementation lives in:

- `mlp/day5_stage2_recent_window/features.py`
- `mlp/day5_stage2_recent_window/mlp_model.py`
- `mlp/day5_stage2_recent_window/self_test_mlp.py`
- `mlp/day5_stage2_recent_window/walk_forward_mlp.py`

## Design

The model predicts `target_5d = close(t+5) / close(t) - 1`.

The validation and test metric is 5-day excess return versus the CSI500
benchmark:

`portfolio_5d_return - benchmark_5d_return`.

The feature set keeps the strongest public-price Stage 1 design:

- short-horizon momentum and reversal: 1/2/3/5/10/20-day returns;
- market-relative returns versus CSI500;
- 5-day relative strength and idiosyncratic momentum;
- intraday gap/range/close-position features;
- volatility, downside volatility, beta, and idiosyncratic volatility;
- volume, amount, turnover, and cross-sectional rank features.

The model search includes:

- lookbacks: `63,126,189,252,full`;
- MLP configs: `narrow_deep`, `narrow_deep_lighter_reg`, `wider_deep`;
- top-k choices: `30,35,40`;
- capped softmax and risk-adjusted softmax weighting.

## Sanity Check Already Done

Without training, the feature pipeline was checked on the current dataset:

- horizon: `5`;
- target: `target_5d`;
- feature count: `55`;
- panel rows: `159,277`;
- training rows: `125,657`;
- latest prediction date: `2026-04-30`;
- prediction rows: `498`.

## Commands

Run canonical self-test:

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_window/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2_recent_window/self_test.json
```

Run walk-forward robustness test:

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_window/walk_forward_mlp.py \
  --json-out submissions/mlp/day5_stage2_recent_window/walk_forward.json
```

Generate a Stage 2 development submission:

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_window/mlp_model.py \
  --out submissions/mlp/day5_stage2_recent_window/submission.csv \
  --json-out submissions/mlp/day5_stage2_recent_window/dev.json
```

Validate CSV format:

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python baseline/validate_submission.py \
  submissions/mlp/day5_stage2_recent_window/submission.csv
```

## What To Watch

Good signs:

- validation and test `mean_excess_return` are positive;
- positive excess rate is at least 50%;
- walk-forward weighted mean excess return is positive;
- selected lookback is not unstable across every split.

Bad signs:

- canonical test is positive but walk-forward is strongly negative;
- validation chooses very aggressive softmax but test has low positive excess
  rate;
- rank IC is near zero or negative across most windows.

