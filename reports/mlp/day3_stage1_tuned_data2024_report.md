# Tuned MLP Stage1 Report on `data_2024`

## Purpose

This experiment is designed to test one question only:

- does extending the historical data window back to `2024-01-01` improve the
  current best stage1 model,
  [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)?

To keep the comparison clean, this variant changes only the data source:

- original model data: `data/`
- extended-history model data: `data_2024/`

Everything else is held fixed:

- stage1 target horizon: `3` trading days
- same feature set
- same MLP search space
- same portfolio construction search
- same portfolio constraints from the course requirement

## Requirement Compliance

This variant still follows the course rules:

- stock universe remains whatever is provided through the project download logic
  and final submission validation
- long-only portfolio
- at least `30` positive-weight names
- per-name cap of `10%`
- weights sum to `1`
- no pretrained model
- no post-deadline data
- train / validation / test with embargo

## Code

- model wrapper:
  [mlp/day3_stage1_tuned_data2024/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_data2024/mlp_model.py)
- self-test wrapper:
  [mlp/day3_stage1_tuned_data2024/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_data2024/self_test_mlp.py)
- features:
  [mlp/day3_stage1_tuned_data2024/features.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_data2024/features.py)
- data path helper:
  [mlp/day3_stage1_tuned_data2024/paths.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_data2024/paths.py)

## Output Locations

When you run this experiment, write artifacts here:

- submission:
  [submissions/mlp/day3_stage1_tuned_data2024/submission.csv](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned_data2024/submission.csv)
- dev summary:
  [submissions/mlp/day3_stage1_tuned_data2024/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned_data2024/dev.json)
- self-test summary:
  [submissions/mlp/day3_stage1_tuned_data2024/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned_data2024/self_test.json)

## What To Compare

Current baseline for this comparison:

- current winner:
  [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)
- current self-test:
  [submissions/mlp/day3_stage1_tuned/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned/self_test.json)

Key question:

- does `data_2024` produce a higher `test.mean_excess_return` than the current
  winner's `+1.744%`?

## Result Placeholder

Fill this after running:

- selected config/top_k/weight:
- validation rank IC:
- validation mean excess return:
- test rank IC:
- test mean excess return:
- test positive excess rate:

## Interpretation Placeholder

Use this section after running to answer:

1. Did longer history help?
2. Did the model become more stable or just different?
3. Does this variant replace the current stage1 winner?
