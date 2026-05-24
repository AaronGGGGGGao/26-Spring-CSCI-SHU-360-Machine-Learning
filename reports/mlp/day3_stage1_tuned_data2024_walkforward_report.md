# Tuned MLP Stage1 Walk-Forward Report on `data_2024`

## Purpose

This report is for a robustness evaluation, not a replacement for the canonical
single self-test split.

Goal:

- test whether the current best stage1 model remains strong when evaluated over
  multiple walk-forward windows on the extended-history `data_2024` dataset

## Methodology

Kept fixed:

- same model family:
  [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)
- same 3-day target horizon
- same feature engineering
- same portfolio construction search
- same course-imposed portfolio constraints

Changed:

- data source: `data_2024/`
- evaluation protocol: multiple non-overlapping walk-forward test windows

Per-window structure:

- expanding train set
- embargo
- validation block
- embargo
- test block

## Code

- walk-forward script:
  [mlp/day3_stage1_tuned_data2024/walk_forward_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_data2024/walk_forward_mlp.py)
- model wrapper:
  [mlp/day3_stage1_tuned_data2024/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_data2024/mlp_model.py)
- self-test wrapper:
  [mlp/day3_stage1_tuned_data2024/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_data2024/self_test_mlp.py)

## Output Location

- walk-forward JSON:
  [submissions/mlp/day3_stage1_tuned_data2024/walk_forward.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned_data2024/walk_forward.json)

## Metrics To Watch

Primary:

- `aggregate.weighted_test_mean_excess_return`

Secondary:

- `aggregate.weighted_test_positive_excess_rate`
- `aggregate.weighted_test_rank_ic`

Also inspect:

- each window's `test_metrics.mean_excess_return`
- whether the selected `top_k` and `weight_method` stay consistent across windows

## Interpretation Template

After running, answer:

1. Is the weighted walk-forward excess return still positive?
2. Is the result close to the canonical winner, or much weaker?
3. Are the per-window results stable, or is performance driven by one window?
4. Does longer history help robustness, even if it does not improve the
   canonical single-split score?
