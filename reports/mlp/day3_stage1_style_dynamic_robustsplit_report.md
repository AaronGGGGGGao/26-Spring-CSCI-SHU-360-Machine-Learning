# Style-Dynamic MLP Robust Split Report

## Split

- validation: `15` trading days
- test: `20` trading days
- embargo: `5` trading days

## Result

- selected config / half-life / allocation policy:
  - `narrow_deep_lighter_reg / 40 / trend_dynamic`
- robust test rank IC: `0.0653`
- robust test mean excess return: `+1.130%`
- robust positive excess rate: `70.0%`

## Comparison

Reference robust baseline:

- [mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py)
- test mean excess return: `+0.684%`
- positive excess rate: `70.0%`

Increment:

- excess return improvement: `+0.446%`
- positive excess rate change: `0.0%`

## Interpretation

This is currently the strongest robust-split result in the project. The main
remaining problem is that the same line under the canonical split is weak, so
it is not yet safe to replace the current submission model without further
refinement.
