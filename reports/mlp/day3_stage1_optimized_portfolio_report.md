# Tuned MLP With Covariance-Aware Portfolio Optimization

## Purpose

This experiment keeps the current stage1 winner's score model fixed and changes
only the portfolio construction layer.

Base alpha model:

- [mlp/day3_stage1_tuned/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_tuned/mlp_model.py)

New portfolio layer:

- alpha score from the tuned MLP
- trailing daily-return covariance matrix
- covariance shrinkage via `LedoitWolf`
- constrained optimization under the course rules

## Requirement Compliance

The experiment keeps all course constraints intact:

- long-only weights
- non-negative weights
- weight sum = `1`
- per-name cap = `10%`
- at least `30` positive-weight names
- no pretrained model
- same train / validation / test methodology with embargo

## Objective

For a selected candidate set of `top_k` names, optimize:

```text
maximize    alpha^T w - lambda * w^T Sigma w
```

subject to:

```text
sum(w) = 1
min_weight <= w_i <= 0.10
```

where:

- `alpha` = current MLP score
- `Sigma` = shrunk covariance matrix from trailing 1-day returns
- `lambda` = risk-aversion parameter

## Search Space

Model side:

- same MLP configurations as the tuned leader

Portfolio side:

- `top_k`: `30, 35, 40`
- covariance lookback: `20, 40, 60`
- risk aversion: `0.5, 1.0, 2.0, 5.0`

## Code

- model:
  [mlp/day3_stage1_optimized_portfolio/mlp_model.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_optimized_portfolio/mlp_model.py)
- self-test:
  [mlp/day3_stage1_optimized_portfolio/self_test_mlp.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_optimized_portfolio/self_test_mlp.py)
- features:
  [mlp/day3_stage1_optimized_portfolio/features.py](/Users/codes/projects/ml_competition/mlp/day3_stage1_optimized_portfolio/features.py)

## Outputs

- submission:
  [submissions/mlp/day3_stage1_optimized_portfolio/submission.csv](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_optimized_portfolio/submission.csv)
- dev:
  [submissions/mlp/day3_stage1_optimized_portfolio/dev.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_optimized_portfolio/dev.json)
- self-test:
  [submissions/mlp/day3_stage1_optimized_portfolio/self_test.json](/Users/codes/projects/ml_competition/submissions/mlp/day3_stage1_optimized_portfolio/self_test.json)

## Metrics To Compare

Primary:

- `test.mean_excess_return`

Secondary:

- `test.positive_excess_rate`
- `test.rank_ic`

Benchmark to beat:

- current tuned MLP stage1:
  `+1.744%` self-test test excess return

## Result Placeholder

Fill after running:

- selected config:
- selected `top_k`:
- selected covariance lookback:
- selected risk aversion:
- validation mean excess return:
- test mean excess return:
- test positive excess rate:
- conclusion vs current tuned MLP:
