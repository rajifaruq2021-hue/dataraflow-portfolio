import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples to simulate
n_samples = 5000

# 1. Generate core reservoir and operating parameters
reservoir_pressure = np.random.uniform(2500, 5000, n_samples) # Pr in psi
# Flowing bottom-hole pressure must be less than reservoir pressure
wf_pressure = reservoir_pressure * np.random.uniform(0.05, 0.95, n_samples) # Pwf in psi

# Petrophysical properties
permeability = np.random.uniform(10, 250, n_samples) # k in md
skin_factor = np.random.uniform(-1, 5, n_samples) # dimensionless skin
water_cut = np.random.uniform(0.0, 0.6, n_samples) # fraction (0 to 60%)
gor = np.random.uniform(200, 1500, n_samples) # Gas-Oil Ratio in scf/stb

# 2. Calculate Maximum Flow Rate (q_max) using a modified Vogel inflow relationship baseline
# Affected by permeability and pressure
q_max = (permeability * reservoir_pressure) / (141.2 * 1.2 * 1.0) * (1 + 0.1 * skin_factor)
q_max = np.clip(q_max, 500, 5000) # Realistic bounds for daily production (stb/day)

# 3. Calculate actual oil flow rate (q) using Vogel's multiphase equation
pressure_ratio = wf_pressure / reservoir_pressure
vogel_multiplier = 1.0 - 0.2 * pressure_ratio - 0.8 * (pressure_ratio ** 2)
actual_flow_rate = q_max * vogel_multiplier

# Introduce minor synthetic field noise (real data is never perfectly smooth)
noise = np.random.normal(0, 15, n_samples)
actual_flow_rate = np.maximum(0, actual_flow_rate + noise)

# 4. Assemble into a structured Pandas DataFrame
df_ipr = pd.DataFrame({
    'Reservoir_Pressure_psi': reservoir_pressure,
    'Flowing_BHP_psi': wf_pressure,
    'Permeability_md': permeability,
    'Skin_Factor': skin_factor,
    'Water_Cut_fraction': water_cut,
    'GOR_scf_stb': gor,
    'Oil_Flow_Rate_stb_day': actual_flow_rate
})

# Display the first few rows of our generated dataset
print(df_ipr.head())

# Save to CSV for the next pipeline step
df_ipr.to_csv('data/simulated_ipr_data.csv', index=False)