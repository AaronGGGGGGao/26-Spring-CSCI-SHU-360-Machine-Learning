# Regime-Aware Tuned MLP Stage1 Robust-Split Report

## Purpose

This experiment evaluates the regime-aware tuned MLP under the stricter
single-window self-test protocol already used to stress-test the current tuned
MLP winner.

The model itself is unchanged relative to the regime-aware version:

- [mlp/day3_stage1_regime_aware/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_regime_aware/mlp_model.py)

What changes is only the evaluation split:

- longer validation block
- longer test block
- same embargo logic

## Requirement Compliance

This still follows all project constraints:

- same public historical data
- same 3-day target
- same long-only constrained portfolio construction
- same minimum `30` names
- same `10%` per-name cap
- same `train / validation / test` methodology with embargo
- no pretrained model

## Split Design

Canonical split:

- validation: `10` trading days
- test: `10` trading days

Robust split here:

- validation: `15` trading days
- test: `20` trading days

## Code

- wrapper:
  [mlp/day3_stage1_regime_aware_robustsplit/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_regime_aware_robustsplit/mlp_model.py)
- robust self-test:
  [mlp/day3_stage1_regime_aware_robustsplit/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_regime_aware_robustsplit/self_test_mlp.py)
- features:
  [mlp/day3_stage1_regime_aware_robustsplit/features.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_regime_aware_robustsplit/features.py)

## Outputs

- submission:
  [submissions/mlp/day3_stage1_regime_aware_robustsplit/submission.csv](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_regime_aware_robustsplit/submission.csv)
- dev:
  [submissions/mlp/day3_stage1_regime_aware_robustsplit/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_regime_aware_robustsplit/dev.json)
- robust self-test:
  [submissions/mlp/day3_stage1_regime_aware_robustsplit/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_regime_aware_robustsplit/self_test.json)

## Metrics To Watch

Primary:

- `test.mean_excess_return`

Secondary:

- `test.positive_excess_rate`
- `test.rank_ic`
- selected `half_life`
- selected `regime_strength`

## Result Placeholder

Fill after running:

- selected config:
- selected `top_k`:
- selected weight method:
- selected half-life:
- selected regime strength:
- validation mean excess return:
- test mean excess return:
- test positive excess rate:
- conclusion:
