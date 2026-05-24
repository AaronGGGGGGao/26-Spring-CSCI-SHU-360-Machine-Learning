# Regime-Aware Tuned MLP Stage1 Report

## Purpose

This experiment keeps the current tuned MLP architecture family and replaces the
static training assumption with a regime-aware training setup.

The motivation is that short-horizon 3-day alpha appears regime-sensitive:

- the canonical 10-day test split was optimistic
- the longer robust split still stayed positive, but at a much lower level
- extending history backward to 2024 diluted performance instead of improving it

So the next rational step is not another generic model family, but a model that
can condition on market state and emphasize recent, regime-relevant samples.

## What Changes Relative to the Current Tuned MLP

Base model preserved:

- [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)

New elements:

1. additional market-regime features
2. recency-aware sample weighting
3. regime-similarity sample weighting

The portfolio constraints remain unchanged and still satisfy the course rules.

## Requirement Compliance

- public historical price/index data only
- no pretrained model
- same stage1 `3-day` target
- same long-only constrained portfolio construction
- same `train / validation / test` methodology
- same `>= 30` names and `<= 10%` per-name cap

## Regime Features

Added features include:

- market 3d / 10d / 20d returns
- market 5d / 10d volatility
- market 5v20 trend
- 20d market drawdown
- short-horizon market breadth
- interactions between stock momentum / beta and market regime

These features are date-level public information and do not introduce leakage.

## Training Logic

Instead of fitting all training samples equally, this model searches over:

- recency half-life
- regime-similarity strength

This lets the model assign higher influence to:

- more recent samples
- samples whose market state resembles the validation/test regime

## Code

- model:
  [mlp/day3_stage1_regime_aware/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_regime_aware/mlp_model.py)
- self-test:
  [mlp/day3_stage1_regime_aware/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_regime_aware/self_test_mlp.py)
- features:
  [mlp/day3_stage1_regime_aware/features.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_regime_aware/features.py)

## Outputs

- submission:
  [submissions/mlp/day3_stage1_regime_aware/submission.csv](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_regime_aware/submission.csv)
- dev:
  [submissions/mlp/day3_stage1_regime_aware/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_regime_aware/dev.json)
- self-test:
  [submissions/mlp/day3_stage1_regime_aware/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_regime_aware/self_test.json)

## Metrics To Watch

Primary:

- `test.mean_excess_return`

Secondary:

- `test.positive_excess_rate`
- `test.rank_ic`
- selected `half_life`
- selected `regime_strength`

## Result Placeholder

Fill after running:

- selected config:
- selected `top_k`:
- selected weight method:
- selected half-life:
- selected regime strength:
- validation mean excess return:
- test mean excess return:
- test positive excess rate:
- conclusion:
