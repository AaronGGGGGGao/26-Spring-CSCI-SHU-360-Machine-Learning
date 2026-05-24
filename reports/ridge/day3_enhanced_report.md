# CSI500 Ridge 3-Day Enhanced Report

## 1. Scope

This document records the second stage1-aligned 3-day Ridge model. Unlike the basic 3-day version, this model changes both:

1. feature engineering, to better reflect short-horizon trading structure
2. portfolio construction, to search for a better mapping from model score to final weights

The goal is to maximize 3-day excess return while staying fully within the project requirements.

## 2. Factors

Implementation:

- feature engineering: [ridge/day3_enhanced/features.py](/Users/codes/projects/ml_competition/ridge/day3_enhanced/features.py)

This version keeps the 3-day target:

- `target_3d = close.shift(-3) / close - 1.0`

It expands the feature set beyond the baseline. The added signals fall into five groups.

### 2.1 Short-horizon return and excess-return signals

- `ret_1d`, `ret_2d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`
- `excess_ret_1d`, `excess_ret_3d`, `excess_ret_5d`, `excess_ret_10d`

These are intended to capture the fast cross-sectional momentum or reversal patterns that matter more over a 3-day holding window.

### 2.2 Intraday and range-based signals

- `intraday_ret`
- `gap_1d`
- `range_1d`
- `close_pos_in_range`

These features try to capture whether the daily bar closed strongly or weakly and whether the next few days may continue or mean-revert.

### 2.3 Volatility and downside-risk signals

- `vol_5d`, `vol_10d`, `vol_20d`
- `downside_vol_20d`
- `vol_term_ratio`

These measure both short-term instability and the shape of the recent volatility curve.

### 2.4 Liquidity and activity signals

- `volume_z_20d`
- `amount_z_20d`
- `turnover_z_20d`
- `turnover_ma_10d`

These are meant to capture unusual participation and short-term trading crowding.

### 2.5 Trend and ranked cross-sectional signals

- `close_over_ma10`, `close_over_ma20`, `close_over_ma60`
- `rsi_6`, `rsi_14`
- `ret_3d_rank`, `ret_10d_rank`
- `excess_ret_3d_rank`
- `volume_z_20d_rank`, `turnover_z_20d_rank`
- `vol_10d_rank`, `range_1d_rank`

These create day-level relative signals, which are useful in a stock-selection setting.

## 3. Model and Portfolio Construction

Implementation:

- training / submission: [ridge/day3_enhanced/ridge_model.py](/Users/codes/projects/ml_competition/ridge/day3_enhanced/ridge_model.py)
- self-test: [ridge/day3_enhanced/self_test_ridge.py](/Users/codes/projects/ml_competition/ridge/day3_enhanced/self_test_ridge.py)

Training pipeline:

1. Build the enhanced 3-day feature panel
2. Standardize inputs
3. Fit `Ridge`
4. Tune over:
   - `alpha in {0.01, 0.1, 1, 10, 100}`
   - `top_k in {30, 40, 50, 60, 80}`
   - `weight_method in {equal, rank, softmax, score, rank_inv_vol}`

Selection criterion:

1. validation mean excess return
2. validation positive excess rate
3. validation rank IC

Weight methods tested:

- `equal`: equal weight
- `rank`: rank-decay weight
- `softmax`: softmax on centered scores
- `score`: positive score-proportional weight
- `rank_inv_vol`: rank weight adjusted by inverse recent volatility

This is the first model in the project where score-to-weight mapping is explicitly tuned rather than held fixed.

## 4. Data and Split

### 4.1 Development split

Current data snapshot:

- raw date range: `2025-01-02` to `2026-04-24`
- train: up to `2026-03-30`
- embargo: 5 trading days
- validation: `2026-04-08` to `2026-04-21`

Outputs:

- development summary: [submissions/ridge/day3_enhanced/dev.json](/Users/codes/projects/ml_competition/submissions/ridge/day3_enhanced/dev.json)
- submission: [submissions/ridge/day3_enhanced/submission.csv](/Users/codes/projects/ml_competition/submissions/ridge/day3_enhanced/submission.csv)

### 4.2 Self-test split

Self-test split:

- train: up to `2026-03-09`
- validation: `2026-03-17` to `2026-03-30`
- test: `2026-04-08` to `2026-04-21`
- embargo: 5 trading days

Output:

- [submissions/ridge/day3_enhanced/self_test.json](/Users/codes/projects/ml_competition/submissions/ridge/day3_enhanced/self_test.json)

## 5. Results

### 5.1 Development split

Selected settings:

- alpha: `0.01`
- top_k: `30`
- weight method: `score`

Development validation results:

- rank IC: `0.0707`
- mean portfolio return: `+2.012%`
- mean benchmark return: `+1.377%`
- mean excess return: `+0.635%`
- positive excess rate: `60.0%`

### 5.2 Self-test

Selected settings from self-test validation:

- alpha: `1.0`
- top_k: `30`
- weight method: `score`

Self-test validation:

- rank IC: `-0.0529`
- mean portfolio return: `-1.060%`
- mean benchmark return: `-1.245%`
- mean excess return: `+0.194%`
- positive excess rate: `60.0%`

Self-test test:

- rank IC: `0.0902`
- mean portfolio return: `+2.650%`
- mean benchmark return: `+1.377%`
- mean excess return: `+1.273%`
- positive excess rate: `70.0%`

## 6. Comparison and Analysis

### 6.1 Against the 3-day basic Ridge

Held-out test comparison:

- 3-day basic test excess return: `+0.883%`
- 3-day enhanced test excess return: `+1.273%`

So the enhanced version improves stage1-style test excess return by `+0.390%`.

### 6.2 Against the 5-day Ridge

Held-out test comparison:

- 5-day Ridge test excess return: `+1.328%`
- 3-day enhanced Ridge test excess return: `+1.273%`

The enhanced 3-day model is close, but still slightly below the earlier 5-day Ridge on this fixed split.

### 6.3 What worked

- Enhanced short-horizon features improved the 3-day model materially relative to the basic 3-day version.
- Tuning the score-to-weight mapping was useful; the best configurations concentrated into 30 names with `score`-proportional weighting.
- Held-out test excess return is clearly better than the 3-day basic model and materially above the baseline 5-day self-test result of `+0.916%`.

### 6.4 What did not work

- Validation rank IC remains modest.
- Development validation excess return is lower than the basic 3-day version, so the model is not uniformly stronger across windows.
- The selected solution is more concentrated, which may create more live-window variance.

### 6.5 Interpretation

This result suggests that for the 3-day task, feature engineering and weight construction matter more than simply shortening the label horizon. The best stage1-style result here comes from:

1. features that explicitly encode short-term return, excess-return, range, and liquidity structure
2. a more concentrated portfolio
3. a score-proportional weight rule rather than the baseline rank-decay rule

That is the strongest 3-day result currently in the project.

## 7. Reproducibility

Development run:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python ridge/day3_enhanced/ridge_model.py --out submissions/ridge/day3_enhanced/submission.csv --json-out submissions/ridge/day3_enhanced/dev.json
```

Self-test:

```bash
cd /Users/codes/projects/ml_competition
/Users/codes/projects/ml_competition/.venv312/bin/python ridge/day3_enhanced/self_test_ridge.py --json-out submissions/ridge/day3_enhanced/self_test.json
```
