# Stage 2 5-Day Model Inventory

This file maps the Stage 1 model attempts to the new Stage 2 5-trading-day
implementations. These are new directories and do not modify the Stage 1 code.

## Model Mapping

- Stage 1 provided XGBoost baseline -> Stage 2 provided XGBoost baseline
  - code: `baseline/day5_stage2/xgboost_model.py`
  - self-test: `baseline/day5_stage2/self_test_xgboost.py`

- Stage 1 enhanced Ridge -> Stage 2 enhanced Ridge
  - code: `ridge/day5_stage2_enhanced/ridge_model.py`
  - self-test: `ridge/day5_stage2_enhanced/self_test_ridge.py`

- Stage 1 XGBoost -> Stage 2 XGBoost
  - code: `xgboost/day5_stage2/xgboost_model.py`
  - self-test: `xgboost/day5_stage2/self_test_xgboost.py`

- Stage 1 base MLP -> Stage 2 base MLP
  - code: `mlp/day5_stage2/mlp_model.py`
  - self-test: `mlp/day5_stage2/self_test_mlp.py`

- Stage 1 tuned MLP -> Stage 2 tuned MLP
  - code: `mlp/day5_stage2_tuned/mlp_model.py`
  - self-test: `mlp/day5_stage2_tuned/self_test_mlp.py`

- Stage 1 LSTM -> Stage 2 LSTM
  - code: `lstm/day5_stage2/lstm_model.py`
  - self-test: `lstm/day5_stage2/self_test_lstm.py`

- Stage 1 style-dynamic MLP -> Stage 2 style-dynamic MLP
  - code: `mlp/day5_stage2_style_dynamic/mlp_model.py`
  - self-test: `mlp/day5_stage2_style_dynamic/self_test_mlp.py`
  - robust split: `mlp/day5_stage2_style_dynamic_robustsplit/self_test_mlp.py`

- Stage 1 recent-window MLP -> Stage 2 recent-window MLP
  - code: `mlp/day5_stage2_recent_window/mlp_model.py`
  - self-test: `mlp/day5_stage2_recent_window/self_test_mlp.py`
  - walk-forward: `mlp/day5_stage2_recent_window/walk_forward_mlp.py`

- Stage 1 recent-window ensemble -> Stage 2 recent-window ensemble
  - code: `mlp/day5_stage2_recent_window_ensemble/mlp_model.py`
  - self-test: `mlp/day5_stage2_recent_window_ensemble/self_test_mlp.py`
  - robust split:
    `mlp/day5_stage2_recent_window_ensemble_robustsplit/self_test_mlp.py`
  - walk-forward:
    `mlp/day5_stage2_recent_window_ensemble/walk_forward_mlp.py`

- Stage 1 optimized portfolio branch -> Stage 2 optimized portfolio branch
  - code: `mlp/day5_stage2_optimized_portfolio/mlp_model.py`
  - self-test: `mlp/day5_stage2_optimized_portfolio/self_test_mlp.py`

- Stage 1 model-switch router -> Stage 2 model-switch router
  - code: `mlp/day5_stage2_model_switch/router_model.py`
  - self-test: `mlp/day5_stage2_model_switch/self_test_router.py`
  - robust split: `mlp/day5_stage2_model_switch/self_test_router_robust.py`
  - walk-forward branch:
    `mlp/day5_stage2_model_switch_walkforward/router_model.py`

- Stage 1 excess-target weighted MLP -> Stage 2 excess-target weighted MLP
  - code: `mlp/day5_stage2_recent_excess_weighted/mlp_model.py`
  - self-test: `mlp/day5_stage2_recent_excess_weighted/self_test_mlp.py`
  - robust split:
    `mlp/day5_stage2_recent_excess_weighted/self_test_mlp_robust.py`
  - walk-forward:
    `mlp/day5_stage2_recent_excess_weighted/walk_forward_mlp.py`

- Stage 2 Transformer sequence model
  - code: `transformer/day5_stage2/transformer_model.py`
  - features: `transformer/day5_stage2/features.py`
  - self-test: `transformer/day5_stage2/self_test_transformer.py`
  - target modes: raw 5-day return and cross-sectional excess-return z-score

## Stage 2 Changes

The Stage 2 branches are not just string replacements from 3-day to 5-day:

- The supervised target is `target_5d`.
- The benchmark return is computed with a 5-trading-day forward CSI500 return.
- Training cutoffs use `FORWARD_HORIZON = 5`, so labels cannot look into
  validation or test periods.
- The recent-window branch searches `top_k = 30,35,40`, because a 5-day horizon
  can benefit from slightly broader portfolios than the Stage 1 default.
- The 5-day feature panel adds 5-day relative strength and 5-day idiosyncratic
  momentum while preserving useful shorter historical windows such as 1-day and
  3-day returns as input signals.
- The excess-target branch trains on `target_excess_5d`, while evaluation uses
  raw realized stock 5-day return minus benchmark 5-day return.
- The Transformer branch uses the same `target_5d` evaluation target and the
  same 5-day embargo, but trains on 20-day per-stock sequences. Its stronger
  variant trains on daily cross-sectional `target_excess_5d` z-scores to better
  align with ranking/selection.

## Verification

Completed checks:

- Python syntax compilation for all new Stage 2 source files passed.
- Main model entrypoints respond to `--help`.
- Feature construction was tested on the current parquet data for:
  `baseline/day5_stage2`, `ridge/day5_stage2_enhanced`,
  `mlp/day5_stage2_recent_window`, `mlp/day5_stage2_style_dynamic`, and
  `mlp/day5_stage2_recent_excess_weighted`.
- Verified feature pipeline outputs:
  - baseline target: `target_5d`;
  - enhanced/recent-window target: `target_5d`;
  - excess-target branch target: `target_excess_5d`;
  - latest prediction rows: 498 to 499 depending on feature availability.
