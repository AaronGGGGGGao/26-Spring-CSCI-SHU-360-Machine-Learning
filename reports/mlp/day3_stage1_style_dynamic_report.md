# Style-Dynamic MLP Report

## Goal

Test whether adding structured style/risk features plus dynamic score-to-weight
policies improves 3-day stage1 excess return relative to the current tuned MLP.

## Canonical self-test

- code: [mlp/day3_stage1_style_dynamic/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_style_dynamic/self_test_mlp.py)
- selected config / half-life / policy:
  - `narrow_deep_lighter_reg / 10 / static_softmax_2.0`
- test rank IC: `0.0389`
- test mean excess return: `+0.308%`
- test positive excess rate: `30.0%`

## Robust self-test

- code: [mlp/day3_stage1_style_dynamic_robustsplit/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_style_dynamic_robustsplit/self_test_mlp.py)
- selected config / half-life / policy:
  - `narrow_deep_lighter_reg / 40 / trend_dynamic`
- test rank IC: `0.0653`
- test mean excess return: `+1.130%`
- test positive excess rate: `70.0%`

## Comparison vs tuned MLP

Canonical baseline:

- [mlp/day3_stage1_tuned/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/self_test_mlp.py)
- test mean excess return: `+1.744%`

Robust baseline:

- [mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py)
- test mean excess return: `+0.684%`

Interpretation:

- canonical split: style-dynamic is materially weaker than tuned MLP
- robust split: style-dynamic is materially stronger than tuned MLP
- this is the first branch that improves the more credible long-horizon held-out
  split by a non-trivial amount, but it does not yet justify replacing the
  canonical submission candidate

## Judgment

This line is not ready to replace the current stage1 submission model.

It is, however, the most promising research direction now because:

- it improves robust excess return from `+0.684%` to `+1.130%`
- it does so without external unstable data
- it suggests the remaining upside is more likely to come from structured
  style/risk features plus dynamic portfolio construction than from trying more
  generic model families
