# CSI500 3-Day XGBoost Baseline Report

## 1. Scope

This document records the stage1-aligned 3-day XGBoost baseline. It keeps the provided baseline model family and the provided baseline feature set, but changes the target horizon from 5 trading days to 3 trading days so the evaluation proxy matches the stage1 live window more closely.

Implementation:

- [baseline/day3/xgboost_model.py](/Users/codes/projects/ml_competition/baseline/day3/xgboost_model.py)
- [baseline/day3/self_test_xgboost.py](/Users/codes/projects/ml_competition/baseline/day3/self_test_xgboost.py)
- [baseline/day3/features.py](/Users/codes/projects/ml_competition/baseline/day3/features.py)

## 2. Factors

This baseline intentionally keeps the original feature set unchanged:

- `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d`
- `vol_20d`
- `volume_z_20d`
- `turnover_ma_20d`
- `close_over_ma20`, `close_over_ma60`
- `rsi_14`
- `ret_5d_rank`, `ret_20d_rank`, `vol_20d_rank`

The only target change is:

- `target_3d = close.shift(-3) / close - 1.0`

Portfolio construction remains the original baseline rank-based top-50 rule.

## 3. Model

Learning algorithm:

- `XGBRegressor`

Training logic:

1. build the 3-day feature panel
2. fit XGBoost on the train split
3. validate on the last 10 dates after a 5-day embargo
4. score the latest cross-section
5. build a legal long-only portfolio

This is not a new idea-heavy model. It is the clean 3-day baseline that makes stage1 comparisons fair.

## 4. Data and Split

### 4.1 Development split

- raw data range: `2025-01-02` to `2026-04-24`
- train: up to `2026-03-30`
- embargo: 5 trading days
- validation: `2026-04-08` to `2026-04-21`

Output:

- [submissions/baseline/day3/submission.csv](/Users/codes/projects/ml_competition/submissions/baseline/day3/submission.csv)

### 4.2 Self-test split

- train: up to `2026-03-09`
- validation: `2026-03-17` to `2026-03-30`
- test: `2026-04-08` to `2026-04-21`
- embargo: 5 trading days

Output:

- [submissions/baseline/day3/self_test.json](/Users/codes/projects/ml_competition/submissions/baseline/day3/self_test.json)

## 5. Results

### 5.1 Development split

- validation rank IC: `0.1051`
- validation mean portfolio return: `+1.945%`
- validation mean benchmark return: `+1.377%`
- validation mean excess return: `+0.569%`
- validation positive excess rate: `100.0%`

### 5.2 Self-test

Validation:

- rank IC: `-0.0432`
- mean portfolio return: `-0.744%`
- mean benchmark return: `-1.245%`
- mean excess return: `+0.501%`

Test:

- rank IC: `0.0984`
- mean portfolio return: `+2.144%`
- mean benchmark return: `+1.377%`
- mean excess return: `+0.767%`
- positive excess rate: `90.0%`

## 6. Interpretation

This 3-day baseline is the correct reference point for stage1. It is methodologically cleaner than comparing a 3-day model against the original 5-day baseline, because the holding horizon is matched.

It also establishes a clear threshold for stage1 model improvement:

- 3-day XGBoost baseline self-test test excess return: `+0.767%`

Any stage1 candidate should be judged against this number first.

## 7. Reproducibility

Development run:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python baseline/day3/xgboost_model.py --out submissions/baseline/day3/submission.csv
```

Self-test:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python baseline/day3/self_test_xgboost.py --json-out submissions/baseline/day3/self_test.json
```
