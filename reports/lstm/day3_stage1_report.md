# Stage1 3-Day LSTM Report

## 1. Scope

This experiment tests a small LSTM sequence model for stage1.

It follows the same methodological constraints as the rest of the project:

- public historical price data only
- stage1-aligned 3-day target
- explicit train / validation / test split
- 5-day embargo between split boundaries
- legal long-only portfolio output

Implementation:

- [lstm/day3_stage1/lstm_model.py](/Users/codes/projects/ml_competition/lstm/day3_stage1/lstm_model.py)
- [lstm/day3_stage1/self_test_lstm.py](/Users/codes/projects/ml_competition/lstm/day3_stage1/self_test_lstm.py)
- [lstm/day3_stage1/features.py](/Users/codes/projects/ml_competition/lstm/day3_stage1/features.py)

Outputs:

- [submissions/lstm/day3_stage1/submission.csv](/Users/codes/projects/ml_competition/submissions/lstm/day3_stage1/submission.csv)
- [submissions/lstm/day3_stage1/dev.json](/Users/codes/projects/ml_competition/submissions/lstm/day3_stage1/dev.json)
- [submissions/lstm/day3_stage1/self_test.json](/Users/codes/projects/ml_competition/submissions/lstm/day3_stage1/self_test.json)

## 2. Model Design

### 2.1 Sequence Input

The model reuses the stronger 3-day feature panel, but instead of one-day tabular input, it uses the last 20 trading observations of each stock as a sequence.

- lookback: `20`
- target: `target_3d`
- output: one score per stock-date

### 2.2 Architecture

A small CPU-friendly LSTM was used:

- `lstm_small`
- `lstm_medium`

This keeps the experiment focused on whether sequence structure helps, not on building a large deep-learning stack.

### 2.3 Portfolio Construction

Portfolio construction remains separate from model training:

1. predict per-stock scores
2. select top-k
3. map scores into legal long-only weights

The selected self-test setting was:

- `top_k = 35`
- `weight_method = score_sq`

## 3. Results

### 3.1 Development Split

Development selection:

- config: `lstm_medium`
- top_k: `30`
- weight: `softmax_2.0`

Development validation:

- rank IC: `0.0838`
- mean portfolio return: `+2.501%`
- mean benchmark return: `+1.377%`
- mean excess return: `+1.124%`

### 3.2 Self-test

Self-test selection:

- config: `lstm_medium`
- top_k: `35`
- weight: `score_sq`

Self-test validation:

- rank IC: `0.0243`
- mean portfolio return: `-1.461%`
- mean benchmark return: `-1.245%`
- mean excess return: `-0.216%`

Self-test test:

- rank IC: `0.0591`
- mean portfolio return: `+2.267%`
- mean benchmark return: `+1.377%`
- mean excess return: `+0.890%`
- positive excess rate: `60.0%`

## 4. Interpretation

This is a better sequence-model result than the earlier GRU line, but it is still not a stage1 leader.

Comparison:

- `mlp/day3_stage1_tuned`: `+1.744%`
- `xgboost/day3_stage1`: `+1.394%`
- `lstm/day3_stage1`: `+0.890%`
- `gru/day3_stage1`: `+0.642%`
- `baseline/day3`: `+0.767%`

Conclusions:

- the sequence-model family is viable
- LSTM improves on the first GRU attempt
- but it still does not beat the current tuned MLP or stage1 XGBoost challenger

So the current stage1 primary recommendation remains the tuned MLP.
