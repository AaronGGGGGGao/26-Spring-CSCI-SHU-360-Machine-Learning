# Recent-Window MLP Report

## Goal

Test whether a shorter recent training window outperforms full-history training
 for the stage1 3-day MLP under the same feature set and portfolio constraints.

The model family is unchanged from the tuned MLP branch. The only change is the
 training sample window:

- search `lookback` over `63 / 126 / 189 / 252 / full`
- keep the same:
  - 3-day target
  - long-only portfolio rules
  - `top_k` and weight-method search
  - embargo logic

## Canonical Self-Test

Split:

- train: up to `2026-03-09`
- validation: `2026-03-17` to `2026-03-30`
- test: `2026-04-08` to `2026-04-21`
- embargo: 5 trading days

Selected setup:

- lookback/config/top_k/weight:
  - `189 (2025-05-28) / wider_deep / 30 / softmax_2.0`
- selected recent train rows/dates:
  - `92,735 / 189`

Validation:

- rank IC: `0.0106`
- mean excess return: `+1.200%`

Test:

- rank IC: `0.0159`
- mean 3d returns (portfolio/benchmark/excess):
  - `+3.903% / +1.377% / +2.526%`
- positive excess rate:
  - `100.0%`

## Robust Self-Test

Split:

- train: up to `2026-02-06`
- validation: `2026-02-24` to `2026-03-16`
- test: `2026-03-24` to `2026-04-21`
- embargo: 5 trading days

Selected setup:

- lookback/config/top_k/weight:
  - `126 (2025-08-05) / narrow_deep_lighter_reg / 30 / softmax_2.0`
- selected recent train rows/dates:
  - `61,832 / 126`

Validation:

- rank IC: `-0.0189`
- mean excess return: `+0.969%`

Test:

- rank IC: `0.0562`
- mean 3d returns (portfolio/benchmark/excess):
  - `+2.030% / +1.226% / +0.804%`
- positive excess rate:
  - `60.0%`

## Comparison vs Full-History Tuned MLP

Canonical:

- recent-window: `+2.526%`
- full-history tuned MLP: `+1.744%`
- improvement: `+0.782%`

Robust:

- recent-window: `+0.804%`
- full-history tuned MLP robust: `+0.684%`
- improvement: `+0.120%`

Reference robust research branch:

- style-dynamic robust: `+1.130%`
- but style-dynamic canonical: `+0.308%`

## Multi-Window Walk-Forward

Additional multi-window self-test on the same `data/` panel:

- window 3:
  - selected lookback/config/top_k/weight:
    - `189 (2025-04-25) / wider_deep / 35 / softmax_2.0`
  - test excess return:
    - `-0.854%`
- window 2:
  - selected lookback/config/top_k/weight:
    - `126 (2025-08-12) / narrow_deep_lighter_reg / 35 / softmax_risk_1.5_0.50`
  - test excess return:
    - `+0.678%`
- window 1:
  - selected lookback/config/top_k/weight:
    - `189 (2025-05-28) / wider_deep / 30 / softmax_2.0`
  - test excess return:
    - `+2.526%`

Aggregate walk-forward result:

- weighted mean excess return:
  - `+0.783%`
- weighted positive excess rate:
  - `66.7%`
- weighted rank IC:
  - `-0.0251`

## Conclusion

This is the first branch that:

- materially improves canonical self-test over the previous submission leader
- stays positive on the longer robust split
- also beats the old tuned-MLP robust baseline

Current interpretation:

- recent-window is the best submission candidate
- style-dynamic remains the strongest robust-only research branch
- the results support the hypothesis that for the 3-day task, older training
  history can dilute the current regime rather than help it
- the canonical uplift is real, but the more credible multi-window strength is
  closer to `+0.783%` than to the single-window peak of `+2.526%`
