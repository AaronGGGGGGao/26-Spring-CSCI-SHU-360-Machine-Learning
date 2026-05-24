# Stage 2 5-Day Self-Test Results

## Scope

I ran the Stage 2 5-trading-day branches that were worth testing first:

- provided 5-day XGBoost baseline;
- enhanced Ridge;
- Stage 2 XGBoost;
- tuned MLP;
- recent-window MLP;
- style-dynamic MLP;
- recent excess-target weighted MLP;
- robust split for style-dynamic MLP;
- robust split for excess-target weighted MLP;
- 3-window walk-forward for recent-window MLP;
- 3-window walk-forward for style-dynamic MLP;
- recent-window + style-dynamic score blend;
- recent-window MLP with a small 5-day feature ablation.

I did not run LSTM, ensemble, or model-switch router in this first pass. They
are heavier and Stage 1 did not show them as stable first-choice models. The
current results are enough to identify the useful Stage 2 directions.

## Canonical Self-Test Split

All canonical self-tests use a chronological 5-day split:

- train: up to `2026-03-11`;
- validation: `2026-03-19` to `2026-04-01`;
- test: `2026-04-10` to `2026-04-23`;
- horizon: 5 trading days;
- embargo: 5 trading days.

The metric below is mean 5-day excess return versus CSI500 on the held-out test
set.

| Model | Test excess | Positive excess rate | Rank IC | Selected setup |
|---|---:|---:|---:|---|
| Provided 5-day XGBoost baseline | `+0.621%` | `70.0%` | `0.1252` | top-50 baseline |
| Enhanced Ridge | `+0.073%` | `40.0%` | `0.0834` | alpha 100, top 80, equal |
| Stage 2 XGBoost | `+1.041%` | `70.0%` | `0.0398` | base, top 30, softmax |
| Tuned MLP | `+1.381%` | `80.0%` | `0.0121` | wider_deep, top 30, softmax_risk_1.5_0.50 |
| Recent-window MLP | `+1.005%` | `80.0%` | `0.0303` | lookback 126, narrow_deep_lighter_reg, top 30, softmax_2.0 |
| Style-dynamic MLP | `+1.934%` | `70.0%` | `0.0883` | narrow_deep, half-life 40, static_softmax_2.0 |
| Recent excess-target weighted MLP | `+1.767%` | `90.0%` | `0.0180` | lookback 189, half-life 20, winsor_zscore, wider_deep, static_30_softmax_1.5 |
| Recent/style score blend | `+1.559%` | `70.0%` | `0.0897` | recent weight 0.40, top 30, softmax_2.0 |
| Recent-window + 5d features | `+2.486%` | `90.0%` | `-0.0337` | lookback 189, narrow_deep, top 30, softmax_2.0 |
| Transformer sequence, raw target | `+0.095%` | `60.0%` | `-0.0153` | transformer_small, top 35, score_sq |
| Transformer sequence, xs excess z target | `-1.523%` | `10.0%` | `-0.0629` | transformer_medium, top 30, softmax_2.0 |

## Robust Checks

The robust split uses:

- train: up to `2026-02-10`;
- validation: `2026-02-26` to `2026-03-18`;
- test: `2026-03-26` to `2026-04-23`;
- test length: 20 trading dates;
- horizon: 5 trading days;
- embargo: 5 trading days.

| Model | Robust test excess | Positive excess rate | Rank IC | Interpretation |
|---|---:|---:|---:|---|
| Style-dynamic MLP | `+0.840%` | `55.0%` | `0.0766` | Still positive, but weaker than canonical |
| Recent excess-target weighted MLP | `-0.591%` | `50.0%` | `0.0612` | Canonical looked strong, but robust split failed |

## Walk-Forward Checks

I ran 3-window walk-forward checks with chronological validation/test windows.
Each window uses a 10-trading-day validation block, a 5-trading-day embargo,
and a 10-trading-day test block. This is the most important stability check
because a single canonical test window can overstate a model that happened to
fit the latest market regime.

### Recent-Window MLP

| Window | Test dates | Selected setup | Test excess | Positive excess rate |
|---|---|---|---:|---:|
| 3 | `2026-03-12` to `2026-03-25` | lookback 189, narrow_deep, top 30, softmax_2.0 | `+0.894%` | `60.0%` |
| 2 | `2026-03-26` to `2026-04-09` | lookback 126, narrow_deep_lighter_reg, top 30, softmax_risk_1.5_0.50 | `+2.874%` | `90.0%` |
| 1 | `2026-04-10` to `2026-04-23` | lookback 126, narrow_deep_lighter_reg, top 30, softmax_2.0 | `+1.005%` | `80.0%` |

Aggregate walk-forward:

- weighted mean portfolio return: `+1.884%`;
- weighted mean benchmark return: `+0.294%`;
- weighted mean excess return: `+1.591%`;
- weighted positive excess rate: `76.7%`;
- weighted rank IC: `0.0042`.

### Style-Dynamic MLP

| Window | Test dates | Selected setup | Test excess | Positive excess rate |
|---|---|---|---:|---:|
| 3 | `2026-03-12` to `2026-03-25` | narrow_deep, half-life 40, breadth_dynamic | `-0.496%` | `50.0%` |
| 2 | `2026-03-26` to `2026-04-09` | wider_deep, half-life 10, breadth_dynamic | `+2.631%` | `70.0%` |
| 1 | `2026-04-10` to `2026-04-23` | narrow_deep, half-life 40, static_softmax_2.0 | `+1.934%` | `70.0%` |

Aggregate walk-forward:

- weighted mean portfolio return: `+1.650%`;
- weighted mean benchmark return: `+0.294%`;
- weighted mean excess return: `+1.356%`;
- weighted positive excess rate: `63.3%`;
- weighted rank IC: `0.0290`.

### Recent/Style Score Blend

This branch trains the recent-window MLP and style-dynamic MLP independently,
z-scores both daily cross-sectional score vectors, and chooses a validation
blend weight. The final portfolio is built once from the blended score rather
than using a heavy model-switch router.

| Window | Test dates | Selected setup | Test excess | Positive excess rate |
|---|---|---|---:|---:|
| 3 | `2026-03-12` to `2026-03-25` | recent weight 0.60, top 30, softmax_2.0 | `+0.211%` | `60.0%` |
| 2 | `2026-03-26` to `2026-04-09` | recent weight 0.60, top 35, softmax_1.8 | `+4.488%` | `80.0%` |
| 1 | `2026-04-10` to `2026-04-23` | recent weight 0.40, top 30, softmax_2.0 | `+1.559%` | `70.0%` |

Aggregate walk-forward:

- weighted mean portfolio return: `+2.379%`;
- weighted mean benchmark return: `+0.294%`;
- weighted mean excess return: `+2.086%`;
- weighted positive excess rate: `70.0%`;
- weighted rank IC: `0.0263`.

### Recent-Window With 5-Day Feature Ablation

This branch keeps the recent-window model and portfolio construction but adds
5-day-horizon features: lagged 5-day stock and excess returns, 5-day amount and
turnover z-scores, 5-day range position, and cross-sectional ranks for the new
5-day signals.

| Window | Test dates | Selected setup | Test excess | Positive excess rate |
|---|---|---|---:|---:|
| 3 | `2026-03-12` to `2026-03-25` | lookback 126, wider_deep, top 30, softmax_1.8 | `+0.350%` | `60.0%` |
| 2 | `2026-03-26` to `2026-04-09` | lookback 189, narrow_deep, top 35, softmax_1.8 | `+1.146%` | `50.0%` |
| 1 | `2026-04-10` to `2026-04-23` | lookback 189, narrow_deep, top 30, softmax_2.0 | `+2.486%` | `90.0%` |

Aggregate walk-forward:

- weighted mean portfolio return: `+1.621%`;
- weighted mean benchmark return: `+0.294%`;
- weighted mean excess return: `+1.327%`;
- weighted positive excess rate: `66.7%`;
- weighted rank IC: `0.0202`.

## Current Interpretation

The best canonical result is now `mlp/day5_stage2_recent_window_5d_features`
with `+2.486%` test excess and `90.0%` positive excess rate. However, its
walk-forward aggregate is only `+1.327%`, below the original recent-window MLP.
This means the new 5-day features help the latest window but are not yet stable
enough to be the main submission choice.

The strongest overall tested direction is
`mlp/day5_stage2_style_recent_blend`. Its canonical result is not the highest
at `+1.559%`, but it has the best 3-window walk-forward aggregate at `+2.086%`.
This is the best evidence so far that the recent-window and style-dynamic alpha
signals are complementary across regimes.

`mlp/day5_stage2_style_dynamic` remains a serious standalone candidate:
canonical excess is `+1.934%`, robust excess is `+0.840%`, and walk-forward
excess is `+1.356%`. It is useful, but the blend is better on walk-forward.

`mlp/day5_stage2_recent_window` remains a reliable fallback. Its canonical
result is lower at `+1.005%`, but its 3-window walk-forward aggregate is strong
at `+1.591%`.

The standalone Transformer branch is not competitive. The raw-return target
version has only `+0.095%` canonical test excess. The cross-sectional
excess-z-score target improves validation excess to `+1.721%`, but fails on the
canonical test window at `-1.523%`, indicating validation overfit under the
current data size. It should stay in the report as an attempted deep sequence
model, not as a submission candidate.

The `mlp/day5_stage2_recent_excess_weighted` branch should not be the current
main model. It has strong canonical test excess (`+1.767%`) and the highest
canonical positive rate (`90.0%`), but its robust test excess is negative
(`-0.591%`).

The Ridge branch is not worth more time now. It barely beats zero excess on the
test window and has only `40.0%` positive excess rate.

## Current Ranking

1. `mlp/day5_stage2_style_recent_blend`: best walk-forward aggregate, best current submission candidate.
2. `mlp/day5_stage2_recent_window`: strongest simple fallback with stable walk-forward evidence.
3. `mlp/day5_stage2_style_dynamic`: strong canonical and positive robust result, but weaker walk-forward than blend.
4. `mlp/day5_stage2_recent_window_5d_features`: best canonical, but walk-forward does not confirm stability.
5. `mlp/day5_stage2_tuned`: simple, strong canonical baseline above XGBoost.
6. `xgboost/day5_stage2`: useful non-neural benchmark, clearly beats provided baseline.
7. `mlp/day5_stage2_recent_excess_weighted`: keep as research branch, not final candidate yet.
8. `transformer/day5_stage2`: attempted sequence model, not competitive on held-out test.
9. `ridge/day5_stage2_enhanced`: stop for now.

## Generated Submission Files

After the self-tests, I generated `submission.csv` for every model run in this
first Stage 2 pass. All generated CSV files passed `baseline/validate_submission.py`.

| Model | Submission file | Rows | Dev validation excess | Dev selected setup |
|---|---|---:|---:|---|
| Provided baseline | `submissions/baseline/day5_stage2/submission.csv` | 50 | -- | top-50 baseline |
| Enhanced Ridge | `submissions/ridge/day5_stage2_enhanced/submission.csv` | 50 | `+0.366%` | alpha 10, top 50, equal |
| Stage 2 XGBoost | `submissions/xgboost/day5_stage2/submission.csv` | 35 | `+1.869%` | base, top 35, score_sq |
| Tuned MLP | `submissions/mlp/day5_stage2_tuned/submission.csv` | 30 | `+2.380%` | narrow_deep, top 30, softmax_1.8 |
| Recent-window MLP | `submissions/mlp/day5_stage2_recent_window/submission.csv` | 30 | `+3.185%` | lookback 189, wider_deep, top 30, softmax_2.0 |
| Style-dynamic MLP | `submissions/mlp/day5_stage2_style_dynamic/submission.csv` | 30 | `+2.145%` | narrow_deep_lighter_reg, half-life 10, breadth_dynamic |
| Excess-target weighted MLP | `submissions/mlp/day5_stage2_recent_excess_weighted/submission.csv` | 40 | `+1.779%` | lookback 189, half-life 20, winsor_zscore, narrow_deep, static_40_softmax_1.8 |
| Recent/style score blend | `submissions/mlp/day5_stage2_style_recent_blend/submission.csv` | 30 | `+2.618%` | recent weight 0.60, top 30, softmax_1.8 |
| Recent-window + 5d features | `submissions/mlp/day5_stage2_recent_window_5d_features/submission.csv` | 30 | `+2.524%` | lookback 189, narrow_deep, top 30, softmax_2.0 |
| Transformer sequence | `submissions/transformer/day5_stage2/submission.csv` | 35 | `-0.321%` | xs_excess_z, transformer_medium, top 35, score_sq |

The highest generated submission by latest dev validation is still
`mlp/day5_stage2_recent_window`, with validation excess `+3.185%`. The better
risk-adjusted choice is now the score blend: its dev validation excess is
lower at `+2.618%`, but its walk-forward test excess is higher at `+2.086%`.

## Recommended Next Step

Use `submissions/mlp/day5_stage2_style_recent_blend/submission.csv` as the
current Stage 2 leading candidate. Keep
`submissions/mlp/day5_stage2_recent_window/submission.csv` as the conservative
fallback, because it has simpler mechanics and still has strong walk-forward
support.

Do not choose `mlp/day5_stage2_recent_window_5d_features` solely from the
canonical result. It needs more ablation work because the latest-window gain did
not translate into the 3-window walk-forward aggregate.

## Reproducibility

The root `README.md` contains the grader-facing reproduction instructions:

- dependency installation with `requirements.txt`;
- required data files and optional data download/update commands;
- exact command to regenerate the selected
  `submissions/mlp/day5_stage2_style_recent_blend/submission.csv`;
- CSV validation command;
- canonical self-test and walk-forward commands;
- fixed random seed and final search/config settings.
