# Recent-Window Excess-Target Weighted MLP Report

To fill after running:

- development result
- canonical self-test result
- robust self-test result
- walk-forward aggregate
- comparison vs `mlp/day3_stage1_recent_window`

This branch keeps the current recent-window feature set, but changes three
things while staying within the original competition constraints:

- trains on `target_excess_3d`
- denoises the supervised target with daily cross-sectional `winsor_zscore` or
  `rank` transforms
- searches both static allocation policies and lightweight market-state
  allocation policies that only switch `top_k` and `weight_method`

The model still produces a standard long-only `stock_code,weight` submission
with at least 30 names, weight sum 1, and max single-name weight 10%.
