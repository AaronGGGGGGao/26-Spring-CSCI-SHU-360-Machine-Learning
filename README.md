# CSI500 Machine Learning Stock Selection

This project is a two-stage machine learning stock-selection system for a
CSI500 portfolio competition. The goal is to predict short-horizon stock
returns, convert model scores into valid long-only portfolios, and evaluate
portfolio excess return against the CSI500 benchmark.

## Course Outcome

- **Stage 1:** full score
- **Stage 2:** A

The final submissions use different models because the two stages have
different prediction horizons:

- **Stage 1:** 3-trading-day return prediction
- **Stage 2:** 5-trading-day return prediction

## Final Selected Models

### Stage 1: Recent-Window MLP

The final Stage 1 model is a **recent-window multi-layer perceptron (MLP)**.
It predicts 3-day forward returns for CSI500 constituents and converts the
scores into a constrained long-only portfolio.

Key design choices:

- Uses only public historical market data: stock OHLCV, volume, amount,
  turnover when available, CSI500 constituents, and CSI500 index data.
- Builds short-horizon technical and cross-sectional features, including
  momentum, market-relative returns, volatility, liquidity, beta,
  idiosyncratic risk, and rank-normalized signals.
- Trains on recent rolling windows instead of always using full history,
  because a 3-day target is sensitive to recent market regimes.
- Selects lookback length, MLP architecture, top-k portfolio size, and weight
  method using validation data only.

Stage 1 final model:

```text
Model: recent-window MLP
Horizon: 3 trading days
Final branch: mlp/day3_stage1_recent_window
Portfolio rule: top-k selection + capped softmax weighting
Constraint: long-only, sum to 1, max 10% per stock, at least 30 holdings
```

### Stage 2: Portfolio-Level Ensemble

The final Stage 2 model is a **portfolio-level ensemble**. Instead of forcing
all model scores onto a single raw scale, each child model first builds a
valid portfolio. The final model then blends portfolio weights.

The main child components are:

- **Recent/style score blend:** combines a recent-window MLP and a
  style-dynamic MLP after daily cross-sectional score standardization.
- **5-day-feature branch:** adds features explicitly aligned with the 5-day
  target, such as lagged 5-day returns, 5-day excess returns, 5-day liquidity
  z-scores, and 5-day range-position features.

Stage 2 final model:

```text
Model: portfolio-level ensemble
Horizon: 5 trading days
Final branch: mlp/day5_stage2_portfolio_ensemble
Portfolio rule: child portfolios -> validation-selected ensemble weights
Constraint: long-only, sum to 1, max 10% per stock, at least 30 holdings
```

## Feature Engineering

All features are derived from public data. No private datasets, news feeds, or
external alternative data are used.

Main feature groups:

- **Recent momentum:** 1, 2, 3, 5, 10, and 20-day returns
- **Market-relative returns:** stock returns minus CSI500 index returns
- **Intraday pressure:** opening gap, intraday return, range, close position
  within the range
- **Volatility and risk:** rolling volatility, downside volatility, beta,
  idiosyncratic volatility, idiosyncratic momentum
- **Liquidity:** standardized volume, standardized amount, turnover z-scores
- **Cross-sectional ranks:** rank-normalized versions of selected momentum,
  liquidity, volatility, beta, and price-location features
- **Stage 2 style/regime features:** market trend, drawdown, breadth,
  dispersion, residual momentum, liquidity shocks, and interaction terms
- **Stage 2 5-day features:** lagged 5-day returns, 5-day excess returns,
  5-day liquidity z-scores, 5-day range, and related ranks

## Model Selection and Validation

The project uses chronological validation to avoid look-ahead bias.

Core validation rules:

- Split data by date, never randomly by row.
- Keep all stocks from the same trading date in the same split.
- Remove rows whose future return target is not observable.
- Use an embargo between train, validation, and test windows.
- Select hyperparameters using validation only.
- Report final performance on held-out test windows.

The main evaluation metric is mean excess return:

```text
portfolio return - CSI500 benchmark return
```

Positive excess rate and rank information coefficient are also monitored, but
final model selection prioritizes held-out excess return and walk-forward
robustness.

## Results Summary

### Stage 1

The recent-window MLP outperformed the provided Stage 1 baseline and received
full score in the course evaluation.

Key self-test results:

```text
Provided 3-day baseline excess: +0.767%
Final recent-window MLP canonical excess: +2.526%
Recent-window MLP robust split excess: +0.804%
Recent-window MLP walk-forward excess: +0.783%
```

### Stage 2

The portfolio-level ensemble was selected because it balanced canonical
performance, walk-forward robustness, and model simplicity. A single branch
had the highest canonical score, but it was less stable across multiple
windows, so it was not selected as the final model.

Key self-test results:

```text
Provided 5-day baseline excess: +0.621%
Portfolio-level ensemble canonical excess: +2.023%
Portfolio-level ensemble walk-forward excess: +2.273%
```

Additional post-cutoff check:

```text
Evaluation window: 2026-05-11 to 2026-05-15
Portfolio return: -0.016%
CSI500 benchmark return: -1.815%
Excess return: +1.800%
```

Stage 2 received an **A** in the course evaluation.

## Reproducing the Final Portfolios

The final submission package contains wrapper scripts for reproducing the two
submitted portfolios.

Recommended workflow:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run Stage 1:

```bash
python stage1_submission/run_stage1.py
```

Run Stage 2:

```bash
python stage2_submission/run_stage2.py
```

The Stage 2 wrapper refreshes data to the week-2 submission cutoff before
generating the final portfolio:

```bash
python download_data.py --update --end 20260510
```

## Repository Structure

```text
baseline/                         Baseline scoring and validation utilities
ridge/                            Ridge feature/model experiments
mlp/day3_stage1_recent_window/    Final Stage 1 model
mlp/day5_stage2_portfolio_ensemble/ Final Stage 2 model
stage1_submission/                Stage 1 reproduction wrapper
stage2_submission/                Stage 2 reproduction wrapper
submissions/                      Generated CSV submissions and JSON summaries
data/                             Price, index, and constituent data
```

Some additional model folders are kept because the final models import shared
feature builders, child models, or validation utilities from earlier
experiments. They are part of the reproducible pipeline, not separate final
submissions.

## Technical Highlights

- Built an end-to-end ML pipeline for data update, feature engineering,
  model training, validation, portfolio construction, and submission checking.
- Compared linear models, tree models, MLPs, sequence models, ranking models,
  score-level blends, and portfolio-level ensembles.
- Used walk-forward validation to reduce reliance on a single favorable test
  window.
- Enforced realistic portfolio constraints: long-only, normalized weights,
  at least 30 holdings, and 10% maximum single-name exposure.
- Kept the final selected models interpretable enough to reproduce and explain
  while still improving materially over the provided baselines.
