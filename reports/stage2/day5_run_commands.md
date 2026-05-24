# Stage 2 5-Day Run Commands

Run commands one block at a time. The heavier MLP/router/LSTM branches can take
substantial time.

## Baseline

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python baseline/day5_stage2/self_test_xgboost.py \
  --json-out submissions/baseline/day5_stage2/self_test.json
```

## Enhanced Ridge

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python ridge/day5_stage2_enhanced/self_test_ridge.py \
  --json-out submissions/ridge/day5_stage2_enhanced/self_test.json
```

## XGBoost

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python xgboost/day5_stage2/self_test_xgboost.py \
  --json-out submissions/xgboost/day5_stage2/self_test.json
```

## Base MLP

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2/self_test.json
```

## Tuned MLP

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_tuned/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2_tuned/self_test.json
```

## Recent-Window MLP

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_window/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2_recent_window/self_test.json
```

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_window/walk_forward_mlp.py \
  --json-out submissions/mlp/day5_stage2_recent_window/walk_forward.json
```

## LSTM

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python lstm/day5_stage2/self_test_lstm.py \
  --json-out submissions/lstm/day5_stage2/self_test.json
```

## Style-Dynamic MLP

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_style_dynamic/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2_style_dynamic/self_test.json
```

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_style_dynamic_robustsplit/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2_style_dynamic_robustsplit/self_test.json
```

## Recent-Window Ensemble

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_window_ensemble/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2_recent_window_ensemble/self_test.json
```

## Optimized Portfolio

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_optimized_portfolio/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2_optimized_portfolio/self_test.json
```

## Excess-Target Weighted MLP

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_excess_weighted/self_test_mlp.py \
  --json-out submissions/mlp/day5_stage2_recent_excess_weighted/self_test.json
```

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_excess_weighted/walk_forward_mlp.py \
  --json-out submissions/mlp/day5_stage2_recent_excess_weighted/walk_forward.json
```

## Model-Switch Router

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_model_switch/self_test_router.py \
  --json-out submissions/mlp/day5_stage2_model_switch/self_test.json
```

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_model_switch_walkforward/walk_forward_router.py \
  --json-out submissions/mlp/day5_stage2_model_switch_walkforward/walk_forward.json
```

## Transformer Sequence Model

The preferred Transformer variant trains on daily cross-sectional excess-return
z-scores, while evaluation still uses realized 5-day portfolio excess return.

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python transformer/day5_stage2/self_test_transformer.py \
  --target-mode xs_excess_z \
  --json-out submissions/transformer/day5_stage2/self_test_xs_excess_z.json
```

Generate its standalone submission candidate:

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python transformer/day5_stage2/transformer_model.py \
  --target-mode xs_excess_z \
  --out submissions/transformer/day5_stage2/submission.csv \
  --json-out submissions/transformer/day5_stage2/dev.json
```

Validate the CSV:

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python baseline/validate_submission.py \
  submissions/transformer/day5_stage2/submission.csv
```

## Generate Candidate Submission

Start with the recent-window MLP candidate:

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python mlp/day5_stage2_recent_window/mlp_model.py \
  --out submissions/mlp/day5_stage2_recent_window/submission.csv \
  --json-out submissions/mlp/day5_stage2_recent_window/dev.json
```

Validate the CSV:

```bash
cd /Users/codes/projects/ml_competition
.venv312/bin/python baseline/validate_submission.py \
  submissions/mlp/day5_stage2_recent_window/submission.csv
```
