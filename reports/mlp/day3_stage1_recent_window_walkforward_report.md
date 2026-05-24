# Recent-Window MLP Walk-Forward Report

## Goal

Evaluate the current best recent-window submission branch across multiple
 historical windows, following the instructor guidance that single-window IC
 and single-window excess return are noisy.

## Window Results

### Window 3

- train end: `2026-01-30`
- validation: `2026-02-09` to `2026-03-02`
- test: `2026-03-10` to `2026-03-23`
- selected lookback/config/top_k/weight:
  - `189 (2025-04-25) / wider_deep / 35 / softmax_2.0`
- test excess return:
  - `-0.854%`

### Window 2

- train end: `2026-02-13`
- validation: `2026-03-03` to `2026-03-16`
- test: `2026-03-24` to `2026-04-07`
- selected lookback/config/top_k/weight:
  - `126 (2025-08-12) / narrow_deep_lighter_reg / 35 / softmax_risk_1.5_0.50`
- test excess return:
  - `+0.678%`

### Window 1

- train end: `2026-03-09`
- validation: `2026-03-17` to `2026-03-30`
- test: `2026-04-08` to `2026-04-21`
- selected lookback/config/top_k/weight:
  - `189 (2025-05-28) / wider_deep / 30 / softmax_2.0`
- test excess return:
  - `+2.526%`

## Aggregate

- weighted test mean excess return:
  - `+0.783%`
- weighted test positive excess rate:
  - `66.7%`
- weighted test rank IC:
  - `-0.0251`

## Comparison

- single-window canonical: `+2.526%`
- single-window robust: `+0.804%`
- multi-window walk-forward aggregate: `+0.783%`

## Conclusion

The recent-window branch survives the multi-window check:

- aggregate excess return remains positive
- it is not just a single-window false positive

But the branch is still regime-sensitive:

- one window is negative
- one window is moderately positive
- one window is very strong

So the credible expected strength is closer to the walk-forward aggregate than
to the best single-window canonical result.
