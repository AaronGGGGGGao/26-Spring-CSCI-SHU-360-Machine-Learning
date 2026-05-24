# Stage1 3-Day Tuned MLP Report

## 1. Scope

This experiment keeps the existing stage1 MLP framework fixed and only performs local tuning around the current winner.

Changes relative to `mlp/day3_stage1`:

1. finer `softmax` temperature search
2. light volatility-aware `softmax` weighting
3. a small architecture neighborhood around the winning `narrow_deep` configuration
4. narrower `top_k` search around the current winner

Implementation:

- [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)
- [mlp/day3_stage1_tuned/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/self_test_mlp.py)
- [mlp/day3_stage1_tuned/features.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/features.py)

Outputs:

- [submissions/mlp/day3_stage1_tuned/submission.csv](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned/submission.csv)
- [submissions/mlp/day3_stage1_tuned/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned/dev.json)
- [submissions/mlp/day3_stage1_tuned/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned/self_test.json)

## 2. Results

### 2.1 Development split

Selected settings:

- config: `narrow_deep_lighter_reg`
- top_k: `30`
- weight method: `softmax_2.0`

Development validation:

- rank IC: `0.0674`
- mean portfolio return: `+3.455%`
- mean benchmark return: `+1.377%`
- mean excess return: `+2.079%`
- positive excess rate: `100.0%`

### 2.2 Self-test

Selected settings from self-test validation:

- config: `narrow_deep`
- top_k: `30`
- weight method: `softmax_2.0`

Self-test validation:

- rank IC: `-0.0074`
- mean portfolio return: `-0.135%`
- mean benchmark return: `-1.245%`
- mean excess return: `+1.110%`

Self-test test:

- rank IC: `0.0243`
- mean portfolio return: `+3.121%`
- mean benchmark return: `+1.377%`
- mean excess return: `+1.744%`
- positive excess rate: `100.0%`

## 3. Comparison

Against the previous `mlp/day3_stage1` leader:

- previous test excess return: `+1.741%`
- tuned test excess return: `+1.744%`

Improvement:

- `+0.003%`

Interpretation:

- The tuned model is better, but only marginally.
- The strongest useful signal from this tuning round is not the average-return gain; it is that `softmax_2.0` and a concentrated 30-name portfolio remain the best region.

## 4. Conclusion

This tuned version becomes the current stage1 leader, but only by a very small margin. It is best treated as a modest refinement of the existing MLP stage1 candidate, not a fundamentally new result.
