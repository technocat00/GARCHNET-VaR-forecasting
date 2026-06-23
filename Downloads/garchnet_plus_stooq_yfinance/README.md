# GARCHNet++: Leverage-Aware VaR and Expected Shortfall Forecasting

This repository implements a course-project extension of the **GARCHNet paper**, where the paper's original model is treated as the primary deep-learning baseline and compared against a leverage-aware improved version.

## Project motivation

The reference paper introduces **GARCHNet** for Value-at-Risk (VaR) forecasting using neural-network-based conditional volatility modelling. In this project, the paper's original GARCHNet setup is reproduced as the **Original GARCHNet baseline**, and then extended into **GARCHNet++** by adding leverage-aware volatility features and Expected Shortfall evaluation.

## Original GARCHNet baseline

In this repository, **Original GARCHNet** refers to the model from the paper before adding our improvements. It uses only past returns as the LSTM input:

```text
[r_{t-p}, ..., r_{t-1}]
```

This baseline is important because it allows us to directly test whether the proposed extensions improve the paper's model rather than only comparing against classical GARCH models.

## Proposed extension: GARCHNet++

GARCHNet++ extends the paper's original model using three leverage-aware features per timestep:

```text
r_t
r_t^2
r_t^2 * I(r_t < 0)
```

These features allow the model to learn volatility magnitude and asymmetric negative-return effects, similar in spirit to GJR-GARCH and EGARCH leverage modelling.

## Core additions beyond the paper

- Leverage-aware input features
- Warm-start friendly rolling-window configuration
- Student-t and skewed-t likelihood losses
- VaR and Expected Shortfall evaluation
- Kupiec UC, Christoffersen CC, DQ, GPL, and ES residual diagnostics

## Suggested model comparison

The final project should compare:

```text
Historical Simulation
GARCH(1,1)
EGARCH
GJR-GARCH
Original GARCHNet from the paper
GARCHNet++ proposed extension
```




## Data fetching

The project fetches index price data automatically instead of storing a bundled dataset.

The paper evaluates daily log returns for:

```text
WIG20
S&P 500
FTSE 100
```

This repository fetches these indices from public data sources where available:

```text
Primary source: Stooq
Fallback source: yfinance
```

The loader converts closing prices into log returns using:

```text
r_t = log(p_t) - log(p_{t-1})
```

Use `data_source = "auto"` in `config.py` to try Stooq first and fall back to yfinance.

## Baseline setup

The repository now includes a unified baseline framework. Every model produces the same forecast panel format:

```text
date, model, alpha, return, sigma2, VaR, ES
```

Implemented baselines:

```text
Historical Simulation
GARCH(1,1)
EGARCH
GJR-GARCH
Original GARCHNet from the paper
GARCHNet++ proposed extension
```

The classical econometric baselines are implemented using the `arch` package.  
The neural baselines use the same PyTorch GARCHNet architecture, with the paper baseline using `input_dim=1` and GARCHNet++ using `input_dim=3`.

Run the baseline experiment using:

```bash
python -m experiments.run_baselines
```

This generates:

```text
results/forecast_panel.csv
results/baseline_summary.csv
```

## Expected outcome

This project aims to show that the proposed **GARCHNet++** extension improves market tail-risk forecasting over both classical econometric baselines and the paper's original GARCHNet model.

Expected measurable outcomes include:

```text
Lower VaR violation error than classical GARCH-type baselines
Improved tail-risk calibration against the paper's original GARCHNet
More robust VaR and ES forecasts during high-volatility market periods
Clear comparison across Historical Simulation, GARCH, EGARCH, GJR-GARCH, Original GARCHNet, and GARCHNet++
```

Resume-style outcome:

```text
Improved tail-risk calibration over 5 baseline models using leverage-aware GARCHNet with VaR and Expected Shortfall backtesting.
```

## Repository structure

```text
config.py
features/
  leverage.py
models/
  garchnet.py
training/
  losses.py
evaluation/
  backtester.py
```

## Resume framing

Built a **leverage-aware GARCHNet++ risk forecasting pipeline** by extending the paper's original GARCHNet model for **1%/2.5%/5% VaR and Expected Shortfall** estimation using hybrid GARCH-LSTM modelling and statistical backtesting.
