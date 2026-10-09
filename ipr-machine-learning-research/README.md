# Machine Learning-Driven Inflow Performance Relationships (IPR)

## Abstract
This project establishes a reproducible, data-driven machine learning framework to model and predict transient oil flow rates in multiphase petroleum reservoirs. Traditional IPR curves rely on empirical correlations (such as Vogel's or Fetkovich's equations) that often fall short under complex fluid properties and transient operating conditions. By leveraging an optimized **XGBoost Regressor** trained on physics-informed synthetic production data ($N = 5,000$), the framework achieves exceptional predictive accuracy ($R^2 = 0.9956$), offering production engineers an instantaneous tool for real-time well optimization.

## Repository Structure
```text
ipr-machine-learning-research/
├── data/
│   └── simulated_ipr_data.csv       # Generated synthetic dataset (5,000 samples)
├── src/
│   ├── generate_ipr_data.py         # Script 1: Dataset simulation pipeline
│   ├── preprocess_data.py           # Script 2: Feature engineering & standardization
│   └── train_model.py               # Script 3: XGBoost training & metric evaluation
├── requirements.txt                 # Python dependencies list
└── README.md                        # Project documentation & references