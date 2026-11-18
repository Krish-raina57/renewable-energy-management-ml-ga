# Methodology

This document summarizes the workflow represented in the original notebook and its accompanying research paper. It is descriptive documentation, not a claim that the implementation exactly reproduces every experiment in the paper.

## Forecasting

### PV power
The supplied PV table contains PV power and weather/solar-related predictors (direct-in-plane irradiance, sun height, air temperature, wind speed), alongside calendar fields. The notebook experiments with Random Forest, XGBoost, and a stacking ensemble.

### Electricity demand
The supplied demand table contains consumption, production, grid import, feed-in, battery charge/discharge, state of charge, and calendar fields. The notebook experiments with Linear Regression, Decision Tree, Gradient Boosting, and stacking.

## Energy management

The notebook connects hourly PV and demand profiles to an EMS simulation. It explores Genetic Algorithm scheduling variants and a MILP-based alternative. The recorded comparison is reported in the main README.

## Important distinction

The source paper describes its own methodology and headline results. The notebook contains its own model choices, preprocessing, optimization implementation, and recorded results. Repository documentation should keep those two evidence sets separate.
