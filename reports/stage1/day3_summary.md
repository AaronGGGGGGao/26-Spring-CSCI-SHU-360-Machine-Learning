# Stage1 3-Day Model Summary

## Goal

This note compares all serious 3-day stage1 candidates built so far under the same held-out self-test split.

Shared self-test split:

- train: up to `2026-03-09`
- validation: `2026-03-17` to `2026-03-30`
- test: `2026-04-08` to `2026-04-21`
- embargo: 5 trading days

## Variants Tested

### 1. 3-day XGBoost baseline

- code: [baseline/day3/xgboost_model.py](/Users/codes/projects/ml_competition/baseline/day3/xgboost_model.py)
- self-test test excess return: `+0.767%`

### 2. 3-day Ridge enhanced

- code: [ridge/day3_enhanced/ridge_model.py](/Users/codes/projects/ml_competition/ridge/day3_enhanced/ridge_model.py)
- self-test test excess return: `+1.273%`

### 3. 3-day XGBoost stage1

- code: [xgboost/day3_stage1/xgboost_model.py](/Users/codes/projects/ml_competition/xgboost/day3_stage1/xgboost_model.py)
- self-test test excess return: `+1.394%`

### 4. 3-day MLP stage1

- code: [mlp/day3_stage1/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1/mlp_model.py)
- self-test test excess return: `+1.741%`

Interpretation:

- this is the strongest 3-day held-out result so far
- it is the first genuinely different model family to beat the tree-based leader

### 5. 3-day tuned MLP stage1

- code: [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)
- self-test test excess return: `+1.744%`

Interpretation:

- this is the current best stage1 result
- it only improves on the earlier MLP by a tiny margin
- the useful conclusion is that `top_k = 30` and sharper `softmax` weighting remain the best local region

### 6. 3-day LSTM stage1

- code: [lstm/day3_stage1/lstm_model.py](/Users/codes/projects/ml_competition/lstm/day3_stage1/lstm_model.py)
- self-test test excess return: `+0.890%`

Interpretation:

- this improves on the earlier GRU attempt
- it beats the 3-day baseline
- but it still does not beat the tuned MLP or stage1 XGBoost

### 7. 3-day tuned MLP robust split

- code: [mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py)
- self-test split:
  - validation: `15` trading days
  - test: `20` trading days
- held-out test excess return: `+0.684%`
- held-out positive excess rate: `70.0%`

Interpretation:

- this is not a new model; it is a stricter evaluation of the current tuned MLP
- the original `+1.744%` canonical result was optimistic because the test block was short
- the tuned MLP remains positive on a longer held-out block, but its more credible strength is materially lower than the canonical peak

### 8. 3-day style-dynamic MLP

- code: [mlp/day3_stage1_style_dynamic/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_style_dynamic/self_test_mlp.py)
- canonical self-test test excess return: `+0.308%`
- code: [mlp/day3_stage1_style_dynamic_robustsplit/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_style_dynamic_robustsplit/self_test_mlp.py)
- robust self-test test excess return: `+1.130%`

Interpretation:

- this is the first branch that clearly improves the stricter robust split
- it does not beat the canonical tuned MLP
- it is therefore not the current submission leader, but it is the strongest
  ongoing research direction

### 9. 3-day recent-window tuned MLP

- code: [mlp/day3_stage1_recent_window/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_recent_window/mlp_model.py)
- canonical self-test test excess return: `+2.526%`
- code: [mlp/day3_stage1_recent_window_robustsplit/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_recent_window_robustsplit/self_test_mlp.py)
- robust self-test test excess return: `+0.804%`
- code: [mlp/day3_stage1_recent_window/walk_forward_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_recent_window/walk_forward_mlp.py)
- multi-window walk-forward aggregate test excess return: `+0.783%`

Interpretation:

- this is the first branch that clearly beats the old canonical tuned MLP
- it also stays above the old tuned-MLP robust baseline on the longer test block
- it supports the hypothesis that for a 3-day horizon, more recent history is
  more useful than full-history training
- this is now the best submission candidate
- the walk-forward result is still positive, which supports the branch
- but the branch is regime-sensitive, so the single-window `+2.526%` should be
  treated as an upper-end outcome rather than the stable expected level

## Comparison Table

| Model | Self-test Test Excess |
|---|---:|
| 3-day XGBoost baseline | `+0.767%` |
| 3-day Ridge enhanced | `+1.273%` |
| 3-day XGBoost stage1 | `+1.394%` |
| 3-day MLP stage1 | `+1.741%` |
| 3-day tuned MLP stage1 | `+1.744%` |
| 3-day LSTM stage1 | `+0.890%` |
| 3-day tuned MLP robust split (20-day test) | `+0.684%` |
| 3-day style-dynamic MLP canonical | `+0.308%` |
| 3-day style-dynamic MLP robust split (20-day test) | `+1.130%` |
| 3-day recent-window MLP canonical | `+2.526%` |
| 3-day recent-window MLP robust split (20-day test) | `+0.804%` |
| 3-day recent-window MLP walk-forward aggregate | `+0.783%` |

## Current Recommendation

The current stage1 leader is:

- [mlp/day3_stage1_recent_window/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_recent_window/mlp_model.py)

Reason:

- it has the highest canonical held-out 3-day self-test excess return so far:
  `+2.526%`
- it exceeds the previous tuned MLP canonical result:
  `+2.526%` vs `+1.744%`
- it remains positive on the stricter robust split and also improves on the old
  tuned-MLP robust baseline:
  `+0.804%` vs `+0.684%`
- it is therefore the strongest balanced submission candidate seen so far

Research leader under stricter evaluation:

- [mlp/day3_stage1_style_dynamic/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_style_dynamic/mlp_model.py)

Reason:

- its canonical result is weak, so it is not ready to replace the submission candidate
- its robust result `+1.130%` is the strongest longer-block held-out result seen so far
- this makes it the best branch to refine further

## Finalist Rolling Check

To avoid overreacting to one self-test split, a lightweight finalist-only rolling comparison was run with fixed finalist settings across 2 historical anchor windows.

Output:

- [submissions/stage1/rolling_finalists.json](/Users/codes/projects/ml_competition/submissions/stage1/rolling_finalists.json)

Average rolling test results:

| Model | Rolling Mean Excess | Rolling Positive Excess Rate | Rolling Rank IC |
|---|---:|---:|---:|
| 3-day MLP stage1 | `+0.360%` | `55.0%` | `0.0310` |
| 3-day XGBoost stage1 | `-0.080%` | `55.0%` | `0.0654` |

Interpretation:

- MLP still leads on realized rolling excess return.
- XGBoost has a stronger rolling rank IC, but weaker realized excess return.
- The margin is smaller than in the single held-out self-test, so stage1 should stay with MLP, but with realistic confidence rather than overconfidence.

## Robustness Adjustment

The most important update after the original summary is:

- tuned MLP canonical self-test: `+1.744%`
- tuned MLP robust split (20-day test): `+0.684%`
- recent-window canonical self-test: `+2.526%`
- recent-window robust split (20-day test): `+0.804%`
- recent-window walk-forward aggregate: `+0.783%`

This means:

- the earlier short canonical split did overstate the old tuned-MLP edge
- but recent-window training still improves both the canonical split and the
  tuned-MLP robust baseline
- the multi-window result remains positive, which supports recent-window as the
  best submission-oriented branch, but also shows the branch is not uniformly
  strong across all windows
- for future work, `+0.804%` is the more useful current submission-level
  robustness baseline to beat, while `+1.130%` remains the strongest
  robust-only research target from style-dynamic

## Recommended Artifacts

Submission candidate:

- [submissions/mlp/day3_stage1_recent_window/submission.csv](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_recent_window/submission.csv)

Supporting results:

- [reports/mlp/day3_stage1_recent_window_report.md](/Users/codes/projects/ml_competition/reports/mlp/day3_stage1_recent_window_report.md)
- [reports/mlp/day3_stage1_recent_window_robustsplit_report.md](/Users/codes/projects/ml_competition/reports/mlp/day3_stage1_recent_window_robustsplit_report.md)
- [reports/mlp/day3_stage1_recent_window_walkforward_report.md](/Users/codes/projects/ml_competition/reports/mlp/day3_stage1_recent_window_walkforward_report.md)
- [submissions/mlp/day3_stage1_recent_window/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_recent_window/dev.json)
- [submissions/mlp/day3_stage1_recent_window/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_recent_window/self_test.json)
- [submissions/mlp/day3_stage1_recent_window_robustsplit/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_recent_window_robustsplit/self_test.json)
- [submissions/mlp/day3_stage1_recent_window/walk_forward.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_recent_window/walk_forward.json)

- [submissions/mlp/day3_stage1_tuned/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned/dev.json)
- [submissions/mlp/day3_stage1_tuned/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_tuned/self_test.json)
- [submissions/mlp/day3_stage1/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1/dev.json)
- [submissions/mlp/day3_stage1/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1/self_test.json)

Previous leader for reference:

- [submissions/xgboost/day3_stage1/dev.json](/Users/codes/projects/ml_competition/submissions/xgboost/day3_stage1/dev.json)
- [submissions/xgboost/day3_stage1/self_test.json](/Users/codes/projects/ml_competition/submissions/xgboost/day3_stage1/self_test.json)
