# Machine Learning-Driven Inflow Performance Relationships (IPR)

## Overview
This repository contains a reproducible machine learning pipeline for predicting multiphase well productivity in petroleum reservoirs using an optimized **XGBoost Regressor**, replacing traditional empirical correlations with high-fidelity data-driven models.

## Project Structure
- `src/generate_ipr_data.py`: Simulates physics-informed synthetic production logs.
- `src/preprocess_data.py`: Handles feature engineering (Pressure Drawdown, Dimensionless Pressure) and standardization.
- `src/train_model.py`: Trains the XGBoost model and evaluates performance metrics ($R^2$, RMSE, MAE).

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt