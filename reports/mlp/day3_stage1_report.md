# Stage1 3-Day MLP Report

## 1. Scope

This experiment tests a genuinely different model family for stage1: a feed-forward neural network trained on the same 3-day feature panel used by the strongest short-horizon tree models.

Implementation:

- [mlp/day3_stage1/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1/mlp_model.py)
- [mlp/day3_stage1/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1/self_test_mlp.py)
- [mlp/day3_stage1/features.py](/Users/codes/projects/ml_competition/mlp/day3_stage1/features.py)

Outputs:

- [submissions/mlp/day3_stage1/submission.csv](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1/submission.csv)
- [submissions/mlp/day3_stage1/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1/dev.json)
- [submissions/mlp/day3_stage1/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1/self_test.json)

## 2. Factors

This model uses the richer public-price feature panel already developed for short-horizon stage1 work:

- short-horizon returns and lagged returns
- market-relative and excess-return features
- intraday, gap, and range signals
- liquidity and turnover anomalies
- beta, idiosyncratic volatility, and idiosyncratic momentum
- cross-sectional rank features

The target is:

- `target_3d = close.shift(-3) / close - 1.0`

## 3. Model

Model family:

- `MLPRegressor` with standardized inputs

Candidate network settings:

1. `small_relu`: `(64,)`, `alpha=1e-3`
2. `medium_relu`: `(128, 64)`, `alpha=1e-3`
3. `narrow_deep`: `(64, 32)`, `alpha=1e-2`

Portfolio search:

- `top_k`: `30`, `35`, `40`, `50`
- weight method:
  - `equal`
  - `rank`
  - `softmax`
  - `score`
  - `score_sq`
  - `score_inv_vol`
  - `rank_inv_vol`

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

### 4.2 Self-test split

- train: up to `2026-03-09`
- validation: `2026-03-17` to `2026-03-30`
- test: `2026-04-08` to `2026-04-21`
- embargo: 5 trading days

## 5. Results

### 5.1 Development split

Selected settings:

- config: `small_relu`
- top_k: `30`
- weight method: `score_sq`

Development validation:

- rank IC: `0.0392`
- mean portfolio return: `+2.990%`
- mean benchmark return: `+1.377%`
- mean excess return: `+1.613%`
- positive excess rate: `80.0%`

### 5.2 Self-test

Selected settings from self-test validation:

- config: `narrow_deep`
- top_k: `30`
- weight method: `softmax`

Self-test validation:

- rank IC: `-0.0074`
- mean portfolio return: `-0.158%`
- mean benchmark return: `-1.245%`
- mean excess return: `+1.087%`

Self-test test:

- rank IC: `0.0243`
- mean portfolio return: `+3.118%`
- mean benchmark return: `+1.377%`
- mean excess return: `+1.741%`
- positive excess rate: `90.0%`

## 6. Comparison

### 6.1 Against the fair 3-day XGBoost baseline

3-day baseline self-test test excess return:

- `+0.767%`

This MLP:

- `+1.741%`

Improvement:

- `+0.974%`

### 6.2 Against the previous stage1 leader

Previous best 3-day XGBoost stage1 self-test test excess return:

- `+1.394%`

This MLP:

- `+1.741%`

Improvement:

- `+0.347%`

### 6.3 Against the tuned MLP refinement

The later tuned MLP refinement:

- [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)

achieved:

- self-test test excess return: `+1.744%`

So this original `mlp/day3_stage1` model is no longer the best MLP variant, but it remains the base version that the tuned model refined.

### 6.4 Finalist rolling comparison

A lightweight finalist-only rolling comparison was run across 2 historical anchor windows using fixed finalist configurations rather than re-running broad search.

Output:

- [submissions/stage1/rolling_finalists.json](/Users/codes/projects/ml_competition/submissions/stage1/rolling_finalists.json)

Average rolling test results:

- `mlp/day3_stage1`
  - mean excess return: `+0.360%`
  - positive excess rate: `55.0%`
  - rank IC: `0.0310`

- `xgboost/day3_stage1`
  - mean excess return: `-0.080%`
  - positive excess rate: `55.0%`
  - rank IC: `0.0654`

Interpretation:

- MLP remains the better stage1 candidate on rolling excess return.
- XGBoost has the stronger average rank IC, but that did not translate into better realized rolling excess return.
- The rolling edge is much smaller than the single-split self-test edge, so the MLP advantage should be treated as real but not large.

## 7. Analysis

### 7.1 What worked

- A genuinely different model family did add value; this is not just another tree variant.
- The best configuration stayed concentrated at `top_k = 30`, which is consistent with the stronger stage1 models.
- `softmax` weighting was better than the more aggressive `score_sq` weighting on held-out self-test.
- The model achieved the highest stage1 held-out test excess return so far.

### 7.2 What did not work

- Rank IC remains low. The model is winning on realized portfolio construction, not on a strong universal ranking metric.
- The best development configuration and the best self-test configuration differ, so time-window sensitivity remains.

### 7.3 Interpretation

This is now the strongest stage1 model in the project because it has the highest 3-day held-out test excess return among all tested models. The result is important because it shows that the current frontier is no longer confined to boosted trees.

## 8. Reproducibility

Development run:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python mlp/day3_stage1/mlp_model.py --out submissions/mlp/day3_stage1/submission.csv --json-out submissions/mlp/day3_stage1/dev.json
```

Self-test:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python mlp/day3_stage1/self_test_mlp.py --json-out submissions/mlp/day3_stage1/self_test.json
```
