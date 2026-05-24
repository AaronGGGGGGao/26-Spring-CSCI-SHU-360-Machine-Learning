# Stage 1 Model Inventory

This file maps the model names used in `reports/stage1/stage1_self_test_report.tex`
to the actual code directories in the repository.

## Report Name -> Code Path

- `Provided 3-day XGBoost baseline`
  - code: `baseline/day3/xgboost_model.py`
  - self-test: `baseline/day3/self_test_xgboost.py`

- `Enhanced Ridge`
  - code: `ridge/day3_enhanced/ridge_model.py`
  - self-test: `ridge/day3_enhanced/self_test_ridge.py`

- `Stage 1 XGBoost`
  - code: `xgboost/day3_stage1/xgboost_model.py`
  - self-test: `xgboost/day3_stage1/self_test_xgboost.py`

- `Base Stage 1 MLP`
  - code: `mlp/day3_stage1/mlp_model.py`
  - self-test: `mlp/day3_stage1/self_test_mlp.py`

- `Tuned MLP`
  - code: `mlp/day3_stage1_tuned/mlp_model.py`
  - self-test: `mlp/day3_stage1_tuned/self_test_mlp.py`
  - robust split: `mlp/day3_stage1_tuned_robustsplit/self_test_mlp.py`

- `LSTM`
  - code: `lstm/day3_stage1/lstm_model.py`
  - self-test: `lstm/day3_stage1/self_test_lstm.py`

- `Style-dynamic MLP`
  - code: `mlp/day3_stage1_style_dynamic/mlp_model.py`
  - self-test: `mlp/day3_stage1_style_dynamic/self_test_mlp.py`
  - robust split: `mlp/day3_stage1_style_dynamic_robustsplit/self_test_mlp.py`

- `Recent-window MLP`
  - code: `mlp/day3_stage1_recent_window/mlp_model.py`
  - self-test: `mlp/day3_stage1_recent_window/self_test_mlp.py`
  - robust split: `mlp/day3_stage1_recent_window_robustsplit/self_test_mlp.py`
  - walk-forward: `mlp/day3_stage1_recent_window/walk_forward_mlp.py`

- `Recent-window ensemble`
  - code: `mlp/day3_stage1_recent_window_ensemble/mlp_model.py`
  - self-test: `mlp/day3_stage1_recent_window_ensemble/self_test_mlp.py`
  - robust split: `mlp/day3_stage1_recent_window_ensemble_robustsplit/self_test_mlp.py`

- `Optimized portfolio branch`
  - code: `mlp/day3_stage1_optimized_portfolio/mlp_model.py`
  - self-test: `mlp/day3_stage1_optimized_portfolio/self_test_mlp.py`

- `Model-switch router`
  - code: `mlp/day3_stage1_model_switch/router_model.py`
  - self-test: `mlp/day3_stage1_model_switch/self_test_router.py`
  - robust split: `mlp/day3_stage1_model_switch/self_test_router_robust.py`
  - extended walk-forward branch:
    `mlp/day3_stage1_model_switch_walkforward/router_model.py`

- `Excess-target weighted MLP`
  - code: `mlp/day3_stage1_recent_excess_weighted/mlp_model.py`
  - self-test: `mlp/day3_stage1_recent_excess_weighted/self_test_mlp.py`
  - robust split:
    `mlp/day3_stage1_recent_excess_weighted/self_test_mlp_robust.py`
  - walk-forward:
    `mlp/day3_stage1_recent_excess_weighted/walk_forward_mlp.py`

## Audit Result

After checking the repository, none of the major model attempts listed in the
self-test report is actually missing. The confusion comes from naming:

- the report uses human-readable experiment names;
- the repository uses implementation directory names.

So there is nothing to restore for the core experiments in the current report.
The code for every cited major branch is still present.
