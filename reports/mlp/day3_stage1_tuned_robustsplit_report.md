# Tuned MLP Stage1 Robust-Split Report

## Purpose

This experiment evaluates the current stage1 winner under a stricter
single-window self-test protocol.

The model itself is unchanged:

- [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)

What changes is only the self-test split:

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

## Split Design

Canonical split used previously:

- validation: `10` trading days
- test: `10` trading days

Robust split here:

- validation: `15` trading days
- test: `20` trading days

The purpose is to reduce short-window noise and check whether the current
winner remains strong on a larger held-out block.

## Code

- wrapper:
  [mlp/day3_stage1_tuned_robustsplit/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_robustsplit/mlp_model.py)
- robust self-test:
  [mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py)
- features:
  [mlp/day3_stage1_tuned_robustsplit/features.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_robustsplit/features.py)

## Outputs

- submission:
  [submissions/mlp/day3_stage1_tuned_robustsplit/submission.csv](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned_robustsplit/submission.csv)
- dev:
  [submissions/mlp/day3_stage1_tuned_robustsplit/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned_robustsplit/dev.json)
- robust self-test:
  [submissions/mlp/day3_stage1_tuned_robustsplit/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned_robustsplit/self_test.json)

## Metrics To Watch

Primary:

- `test.mean_excess_return`

Secondary:

- `test.positive_excess_rate`
- `test.rank_ic`

Main question:

- does the current tuned MLP remain strong when the held-out test block is made
  materially larger?

## Results

- selected config: `wider_deep`
- selected `top_k`: `35`
- selected weight method: `softmax_1.0`
- validation rank IC: `-0.0936`
- validation mean excess return: `-0.026%`
- test rank IC: `0.0610`
- test mean excess return: `+0.684%`
- test positive excess rate: `70.0%`

## Interpretation

This result changes the confidence level around the earlier canonical winner.

- The tuned MLP remains a positive-alpha model on a larger held-out block.
- However, the earlier `+1.744%` canonical self-test clearly overstated its
  strength.
- A more credible estimate of the model's stage1 performance is now closer to
  the `+0.684%` to `+1.0%` range than the original short-window peak.

## Conclusion

The current tuned MLP is still the leading stage1 candidate, but it should be
treated as a moderate rather than dominant edge. Future improvements should be
judged against this stricter robustness baseline, not only against the original
10-day test split.
