# CSI500 Ridge Model Report

## 1. Scope

This document records the first model built beyond the provided XGBoost baseline: a Ridge regression model using the same provided data, the same feature set, the same portfolio construction rule, and the same split logic. The goal is to make the comparison to the baseline fair and methodologically defensible.

## 2. Factors

The Ridge model uses exactly the same input data and features as the baseline:

- data source: [prices.parquet](/Users/codes/projects/ml_competition/data/prices.parquet)
- feature engineering: [baseline/features.py](/Users/codes/projects/ml_competition/baseline/features.py)

Feature columns:

- `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d`
- `vol_20d`
- `volume_z_20d`
- `turnover_ma_20d`
- `close_over_ma20`, `close_over_ma60`
- `rsi_14`
- `ret_5d_rank`, `ret_20d_rank`, `vol_20d_rank`

The target is unchanged from the baseline:

- `target_5d = close.shift(-5) / close - 1.0`

This preserves comparability. At this stage, the only substantive change is the prediction model itself.

## 3. Model

### 3.1 Learning algorithm

The 5-day Ridge model is now organized under [ridge/day5/ridge_model.py](/Users/codes/projects/ml_competition/ridge/day5/ridge_model.py).

The training pipeline is:

1. Standardize the feature columns with `StandardScaler`
2. Fit a `Ridge` regression model on the training set
3. Tune the regularization parameter `alpha` on the validation set

Candidate alphas:

- `0.01`
- `0.1`
- `1.0`
- `10.0`
- `100.0`

Validation selection criterion:

- primary: validation mean excess return
- tie-breaker: validation rank IC

This is a reasonable choice because the competition ultimately ranks submissions by excess return rather than by MSE.

### 3.2 Portfolio construction

Portfolio construction is intentionally unchanged from the baseline:

1. Predict a score for each stock
2. Rank stocks by predicted score
3. Select the top 50 names
4. Assign rank-based weights
5. Enforce the 10% cap and redistribute excess weight

This means the comparison isolates the effect of the model choice rather than mixing model changes with portfolio-rule changes.

## 4. Data and Split

### 4.1 Development split

The standard development run uses the same split as the baseline:

- raw data range: `2025-01-02` to `2026-04-24`
- train: up to `2026-03-26`
- embargo: 5 trading days
- validation: from `2026-04-03` to `2026-04-17`

### 4.2 Self-test split

For the 25% self-test requirement, a separate script was created:

- [ridge/day5/self_test_ridge.py](/Users/codes/projects/ml_competition/ridge/day5/self_test_ridge.py)

It uses:

```text
[ train ][ embargo ][ validation ][ embargo ][ test ]
```

Current split:

- train: up to `2026-03-05`
- validation: `2026-03-13` to `2026-03-26`
- test: `2026-04-03` to `2026-04-17`
- embargo: 5 trading days on both boundaries

Outputs:

- development summary: [submissions/ridge/day5/dev.json](/Users/codes/projects/ml_competition/submissions/ridge/day5/dev.json)
- self-test summary: [submissions/ridge/day5/self_test.json](/Users/codes/projects/ml_competition/submissions/ridge/day5/self_test.json)
- current legal submission: [submissions/ridge/day5/submission.csv](/Users/codes/projects/ml_competition/submissions/ridge/day5/submission.csv)

## 5. Results

### 5.1 Development split results

Ridge:

- selected alpha: `1.0`
- validation rank IC: `0.1260`
- validation mean portfolio return: `+4.834%`
- validation mean benchmark return: `+3.270%`
- validation mean excess return: `+1.564%`
- validation positive excess rate: `60.0%`

Baseline on the same development split:

- validation rank IC: `0.1814`
- validation mean excess return: `+1.596%`
- validation positive excess rate: `100.0%`

Comparison:

- Ridge is slightly worse than baseline on development validation excess return (`+1.564%` vs `+1.596%`)
- Ridge is also worse on development rank IC (`0.1260` vs `0.1814`)

So on the ordinary baseline development split, Ridge does **not** beat the provided XGBoost baseline.

### 5.2 Self-test results

Ridge self-test validation:

- selected alpha: `0.1`
- validation rank IC: `-0.0833`
- validation mean portfolio return: `-2.474%`
- validation mean benchmark return: `-2.371%`
- validation mean excess return: `-0.103%`
- validation positive excess rate: `50.0%`

Ridge self-test test:

- test rank IC: `0.1639`
- test mean portfolio return: `+4.598%`
- test mean benchmark return: `+3.270%`
- test mean excess return: `+1.328%`
- test positive excess rate: `70.0%`

Baseline self-test test:

- test rank IC: `0.1832`
- test mean excess return: `+0.916%`
- test positive excess rate: `90.0%`

Comparison on test:

- Ridge has **higher test excess return** than baseline (`+1.328%` vs `+0.916%`)
- Ridge has **lower test rank IC** than baseline (`0.1639` vs `0.1832`)
- Ridge has **lower positive excess rate** than baseline (`70.0%` vs `90.0%`)

## 6. Analysis

### 6.1 What worked

- The Ridge model is fully compliant with the project requirements.
- It uses the same provided dataset and the same no-leakage split design.
- The generated submission file is valid under the competition constraints.
- Most importantly for the self-test requirement, Ridge improves the baseline's held-out **test excess return** on the current fixed split:
  - Ridge: `+1.328%`
  - Baseline: `+0.916%`

If the course staff interprets "exceeds the provided baseline" primarily through test excess return, this is a meaningful improvement.

### 6.2 What did not work

- Ridge does not beat the baseline on the ordinary development validation run.
- Ridge also underperforms the baseline on test rank IC and positive excess-rate stability.
- The self-test validation window is weak for Ridge: both rank IC and excess return are poor.

This means Ridge is not uniformly better. It improves one important held-out metric, but not the full profile.

### 6.3 Interpretation

The result suggests that a simpler linear model may produce a portfolio with better aggregate excess return over one held-out test window, even though its stock-level ranking quality is weaker than XGBoost's. In other words:

- XGBoost appears stronger as a general cross-sectional ranking model
- Ridge appears capable of producing a competitive portfolio on the current test split

The main limitation is instability across windows. The difference between:

- development validation
- self-test validation
- self-test test

is large enough that neither model should be treated as robust without additional rolling-window experiments.

## 7. Self-test Requirement

Yes, the self-test is required, and it is now implemented for Ridge.

What was needed:

1. a proper `train / validation / test` split
2. embargo gaps to prevent forward-return leakage
3. held-out test metrics reported separately from validation metrics
4. direct comparison against the provided baseline

The Ridge self-test satisfies those conditions through:

- [ridge/day5/self_test_ridge.py](/Users/codes/projects/ml_competition/ridge/day5/self_test_ridge.py)
- [submissions/ridge/day5/self_test.json](/Users/codes/projects/ml_competition/submissions/ridge/day5/self_test.json)

## 8. Reproducibility

Run the development version:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python ridge/day5/ridge_model.py --out submissions/ridge/day5/submission.csv --json-out submissions/ridge/day5/dev.json
```

Run the self-test:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python ridge/day5/self_test_ridge.py --json-out submissions/ridge/day5/self_test.json
```
