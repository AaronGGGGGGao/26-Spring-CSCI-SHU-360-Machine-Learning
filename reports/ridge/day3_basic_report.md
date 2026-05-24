# CSI500 Ridge 3-Day Basic Report

## 1. Scope

This document records the first stage1-aligned 3-day Ridge model. It keeps the same feature engineering and the same portfolio construction rule as the provided baseline and the earlier 5-day Ridge model. The only substantive modeling change is the prediction horizon:

- baseline / prior Ridge: predict 5-day return
- this model: predict 3-day return

This isolates whether horizon alignment alone improves stage1-style excess return.

## 2. Factors

This version uses the same raw data and the same feature set as the provided baseline:

- price data: [prices.parquet](/Users/codes/projects/ml_competition/data/prices.parquet)
- feature engineering: [ridge/day3_basic/features.py](/Users/codes/projects/ml_competition/ridge/day3_basic/features.py)

Feature columns:

- `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d`
- `vol_20d`
- `volume_z_20d`
- `turnover_ma_20d`
- `close_over_ma20`, `close_over_ma60`
- `rsi_14`
- `ret_5d_rank`, `ret_20d_rank`, `vol_20d_rank`

The only target change is:

- `target_3d = close.shift(-3) / close - 1.0`

This is the stage1-aligned target.

## 3. Model

Implementation:

- training / submission: [ridge/day3_basic/ridge_model.py](/Users/codes/projects/ml_competition/ridge/day3_basic/ridge_model.py)
- self-test: [ridge/day3_basic/self_test_ridge.py](/Users/codes/projects/ml_competition/ridge/day3_basic/self_test_ridge.py)

Training pipeline:

1. Standardize feature columns with `StandardScaler`
2. Fit `Ridge`
3. Tune `alpha` over `0.01, 0.1, 1, 10, 100`
4. Select the best model by validation mean excess return, then rank IC

Portfolio construction is intentionally unchanged from baseline:

1. Predict a score for each stock
2. Rank stocks by predicted score
3. Select top 50 names
4. Apply the same rank-based weight rule
5. Enforce the 10% cap and redistribute if needed

This keeps the comparison clean: horizon changes, but features and weight logic do not.

## 4. Data and Split

### 4.1 Development split

Using the current data snapshot:

- raw date range: `2025-01-02` to `2026-04-24`
- train: up to `2026-03-30`
- embargo: 5 trading days
- validation: `2026-04-08` to `2026-04-21`

Output files:

- development summary: [submissions/ridge/day3_basic/dev.json](/Users/codes/projects/ml_competition/submissions/ridge/day3_basic/dev.json)
- submission: [submissions/ridge/day3_basic/submission.csv](/Users/codes/projects/ml_competition/submissions/ridge/day3_basic/submission.csv)

### 4.2 Self-test split

For the 25% self-test requirement:

- train: up to `2026-03-09`
- validation: `2026-03-17` to `2026-03-30`
- test: `2026-04-08` to `2026-04-21`
- embargo: 5 trading days on both boundaries

Output:

- [submissions/ridge/day3_basic/self_test.json](/Users/codes/projects/ml_competition/submissions/ridge/day3_basic/self_test.json)

## 5. Results

### 5.1 Development split

Selected settings:

- alpha: `0.01`

Development validation results:

- rank IC: `0.0829`
- mean portfolio return: `+2.411%`
- mean benchmark return: `+1.377%`
- mean excess return: `+1.034%`
- positive excess rate: `60.0%`

### 5.2 Self-test

Selected settings from self-test validation:

- alpha: `100`

Self-test validation:

- rank IC: `-0.0618`
- mean portfolio return: `-1.418%`
- mean benchmark return: `-1.245%`
- mean excess return: `-0.173%`
- positive excess rate: `50.0%`

Self-test test:

- rank IC: `0.1113`
- mean portfolio return: `+2.260%`
- mean benchmark return: `+1.377%`
- mean excess return: `+0.883%`
- positive excess rate: `50.0%`

## 6. Comparison and Analysis

### 6.1 Against the 5-day Ridge

The 3-day basic model is worse than the earlier 5-day Ridge on held-out test excess return:

- 5-day Ridge test excess return: `+1.328%`
- 3-day basic Ridge test excess return: `+0.883%`

So horizon alignment alone did not improve test excess return in this fixed split.

### 6.2 What worked

- The implementation is fully compliant with project requirements.
- The stage1 horizon is matched directly in the target.
- The submission file is valid.
- The model remains simple and methodologically clean.

### 6.3 What did not work

- The self-test validation window is weak.
- The held-out test excess return is positive, but lower than the 5-day Ridge and lower than the 3-day enhanced version.
- Rank IC is modest, which suggests limited cross-sectional separation power.

### 6.4 Interpretation

This result suggests that simply changing `target_5d` to `target_3d`, while leaving both features and weight construction unchanged, is not enough to maximize stage1-style excess return. The model likely needs either better short-horizon features, a better weight rule, or both.

## 7. Reproducibility

Development run:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python ridge/day3_basic/ridge_model.py --out submissions/ridge/day3_basic/submission.csv --json-out submissions/ridge/day3_basic/dev.json
```

Self-test:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python ridge/day3_basic/self_test_ridge.py --json-out submissions/ridge/day3_basic/self_test.json
```
