import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset we just generated
df = pd.read_csv('data/simulated_ipr_data.csv')

# 2. Engineer Domain-Specific Features
# Pressure Drawdown
df['Pressure_Drawdown_psi'] = df['Reservoir_Pressure_psi'] - df['Flowing_BHP_psi']

# Dimensionless Pressure Ratio
df['Dimensionless_Pressure'] = df['Flowing_BHP_psi'] / df['Reservoir_Pressure_psi']

# Re-order/Select features (X) and target variable (y)
X = df[[
    'Reservoir_Pressure_psi', 
    'Flowing_BHP_psi', 
    'Permeability_md', 
    'Skin_Factor', 
    'Water_Cut_fraction', 
    'GOR_scf_stb', 
    'Pressure_Drawdown_psi', 
    'Dimensionless_Pressure'
]]
y = df['Oil_Flow_Rate_stb_day']

# 3. Split into Train (80%) and Test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete!")
print(f"Training features shape: {X_train_scaled.shape}")
print(f"Testing features shape: {X_test_scaled.shape}")
