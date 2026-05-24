# Recent-Window MLP Robust-Split Report

## Goal

Evaluate whether the recent-window MLP still holds up under the longer, stricter
 stage1 split:

- validation: `15` trading days
- test: `20` trading days
- embargo: `5` trading days

The model family and feature set stay aligned with the tuned MLP branch. The
 only change is restricting training to recent history and searching the
 training lookback window.

## Split

- train: up to `2026-02-06`
- validation: `2026-02-24` to `2026-03-16`
- test: `2026-03-24` to `2026-04-21`
- embargo: `5` trading days

## Selected Setup

- lookback/config/top_k/weight:
  - `126 (2025-08-05) / narrow_deep_lighter_reg / 30 / softmax_2.0`
- selected recent train rows/dates:
  - `61,832 / 126`

## Validation

- rank IC: `-0.0189`
- mean 3d returns (portfolio/benchmark/excess):
  - `-0.198% / -1.167% / +0.969%`

## Held-Out Test

- rank IC: `0.0562`
- mean 3d returns (portfolio/benchmark/excess):
  - `+2.030% / +1.226% / +0.804%`
- positive excess rate:
  - `60.0%`

## Comparison

Against the old full-history tuned MLP robust split:

- recent-window robust: `+0.804%`
- tuned robust: `+0.684%`
- improvement: `+0.120%`

Against the strongest robust-only research branch:

- style-dynamic robust: `+1.130%`
- recent-window robust: `+0.804%`

Interpretation:

- recent-window does not beat the style-dynamic robust research line
- but it does beat the old tuned-MLP robust baseline
- combined with its much stronger canonical result, it remains the best
  submission-oriented branch overall
