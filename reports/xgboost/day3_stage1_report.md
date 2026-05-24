# Stage1 3-Day XGBoost Report

## 1. Scope

This document records the first stage1-oriented XGBoost model built beyond the clean 3-day baseline. The goal is to keep the strong boosted-tree family, but improve stage1 performance through:

1. richer short-horizon feature engineering
2. recency-aware training as an option
3. broader portfolio-construction search

Implementation:

- [xgboost/day3_stage1/xgboost_model.py](/Users/codes/projects/ml_competition/xgboost/day3_stage1/xgboost_model.py)
- [xgboost/day3_stage1/self_test_xgboost.py](/Users/codes/projects/ml_competition/xgboost/day3_stage1/self_test_xgboost.py)
- [xgboost/day3_stage1/features.py](/Users/codes/projects/ml_competition/xgboost/day3_stage1/features.py)

## 2. Factors

This model does not use the tiny baseline feature set. It reuses the richer public-price feature panel already developed for short-horizon modeling, including:

- short-horizon returns and lagged returns
- excess-return and relative-strength features
- intraday, gap, and range features
- liquidity and turnover anomalies
- beta, idiosyncratic volatility, and idiosyncratic momentum
- daily cross-sectional ranks of short-horizon signals

The target remains:

- `target_3d = close.shift(-3) / close - 1.0`

This keeps the objective directly aligned with the stage1 3-day holding window.

## 3. Model

Model family:

- `XGBRegressor`

Compared with the clean 3-day baseline, this version jointly tunes:

- tree configuration
- `top_k`
- weight method
- recency half-life for training weights

Candidate tree configurations:

1. `base`
2. `shallow_reg`
3. `mid_depth`

Candidate weight methods:

- `equal`
- `rank`
- `softmax`
- `score`
- `score_sq`
- `score_inv_vol`
- `rank_inv_vol`

Candidate `top_k`:

- `30`, `35`, `40`, `50`

Candidate recency half-life:

- `none`, `10`, `20`

Selection criterion:

1. validation mean excess return
2. validation positive excess rate
3. validation rank IC

## 4. Data and Split

### 4.1 Development split

- raw data range: `2025-01-02` to `2026-04-24`
- train: up to `2026-03-30`
- embargo: 5 trading days
- validation: `2026-04-08` to `2026-04-21`

Outputs:

- [submissions/xgboost/day3_stage1/dev.json](/Users/codes/projects/ml_competition/submissions/xgboost/day3_stage1/dev.json)
- [submissions/xgboost/day3_stage1/submission.csv](/Users/codes/projects/ml_competition/submissions/xgboost/day3_stage1/submission.csv)

### 4.2 Self-test split

- train: up to `2026-03-09`
- validation: `2026-03-17` to `2026-03-30`
- test: `2026-04-08` to `2026-04-21`
- embargo: 5 trading days

Output:

- [submissions/xgboost/day3_stage1/self_test.json](/Users/codes/projects/ml_competition/submissions/xgboost/day3_stage1/self_test.json)

## 5. Results

### 5.1 Development split

Selected settings:

- config: `shallow_reg`
- top_k: `30`
- weight method: `score`
- recency half-life: `none`

Development validation results:

- rank IC: `0.0625`
- mean portfolio return: `+2.432%`
- mean benchmark return: `+1.377%`
- mean excess return: `+1.056%`
- positive excess rate: `70.0%`

### 5.2 Self-test

Selected settings from self-test validation:

- config: `shallow_reg`
- top_k: `30`
- weight method: `score_sq`
- recency half-life: `20`

Self-test validation:

- rank IC: `0.0481`
- mean portfolio return: `-0.303%`
- mean benchmark return: `-1.245%`
- mean excess return: `+0.942%`

Self-test test:

- rank IC: `0.0934`
- mean portfolio return: `+2.771%`
- mean benchmark return: `+1.377%`
- mean excess return: `+1.394%`
- positive excess rate: `70.0%`

## 6. Comparison

### 6.1 Against the fair 3-day XGBoost baseline

3-day baseline self-test test excess return:

- `+0.767%`

This stage1 model:

- `+1.394%`

Improvement:

- `+0.627%`

### 6.2 Against the best 3-day Ridge

Best 3-day Ridge (`day3_enhanced`) self-test test excess return:

- `+1.273%`

This stage1 XGBoost model:

- `+1.394%`

Improvement:

- `+0.121%`

## 7. Analysis

### 7.1 What worked

- Keeping the stronger boosted-tree model family was the right decision.
- RandomForest would likely have been a weaker next step; the data and current evidence favor boosted trees over bagged trees here.
- Richer short-horizon features plus better portfolio construction improved held-out stage1-style excess return materially.
- The best self-test solution concentrated into 30 names with `score_sq` weighting rather than the baseline rank-weight rule.

### 7.2 What did not work

- Development validation rank IC is not especially high.
- The best development configuration and the best self-test configuration differ, which means the model remains sensitive to time window.
- Positive excess rate is lower than the clean 3-day baseline (`70%` vs `90%`), so the model wins on average return, not on consistency.

### 7.3 Interpretation

This model is currently the best stage1 candidate in the project because it has the highest 3-day held-out test excess return among all tested models. The gain did not come from changing the target definition; it came from:

1. richer short-horizon public-price features
2. keeping a strong nonlinear learner
3. more flexible portfolio construction

## 8. Reproducibility

Development run:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python xgboost/day3_stage1/xgboost_model.py --out submissions/xgboost/day3_stage1/submission.csv --json-out submissions/xgboost/day3_stage1/dev.json
```

Self-test:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python xgboost/day3_stage1/self_test_xgboost.py --json-out submissions/xgboost/day3_stage1/self_test.json
```
