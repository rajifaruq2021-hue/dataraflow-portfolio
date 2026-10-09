import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load, engineer features, and split (same as before to keep script self-contained)
df = pd.read_csv('data/simulated_ipr_data.csv')
df['Pressure_Drawdown_psi'] = df['Reservoir_Pressure_psi'] - df['Flowing_BHP_psi']
df['Dimensionless_Pressure'] = df['Flowing_BHP_psi'] / df['Reservoir_Pressure_psi']

X = df[[
    'Reservoir_Pressure_psi', 'Flowing_BHP_psi', 'Permeability_md', 
    'Skin_Factor', 'Water_Cut_fraction', 'GOR_scf_stb', 
    'Pressure_Drawdown_psi', 'Dimensionless_Pressure'
]]
y = df['Oil_Flow_Rate_stb_day']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Initialize and Train the XGBoost Regressor
print("Training the XGBoost model...")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# 3. Predict and Evaluate
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"R2 Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} stb/day")
print(f"Mean Absolute Error (MAE): {mae:.2f} stb/day")