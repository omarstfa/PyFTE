

# import itertools
# import pandas as pd

# # Simulate all states of 6 components
# columns = ['PV', 'INV', 'BAT', 'CTRL', 'DCL', 'ACL']
# states = list(itertools.product([0, 1], repeat=6))
# df = pd.DataFrame(states, columns=columns)

# # Define the Top Event condition
# def evaluate_TE(row):
#     # Example logic - tweak this based on your real-world knowledge
#     if row['PV'] == 1:
#         return 1
#     if row['INV'] == 1:
#         return 1
#     if row['BAT'] == 1 and row['CTRL'] == 1:
#         return 1
#     if row['DCL'] == 1 and row['ACL'] == 1:
#         return 1
#     return 0

# df['TE'] = df.apply(evaluate_TE, axis=1)
# df.to_csv("pv_system_truth_table.txt", sep=" ", index=False)


#%%
import pandas as pd
import numpy as np

np.random.seed(42)  # Reproducible

# Components
components = ['Sun', 'PV Module', 'Circuit Breaker', 'Diode', 'Fuse', 'Cable']

# Simulate 100 system states (randomly, but not uniformly)
n_samples = 100
data = pd.DataFrame(columns=components)

# Simulate based on likely failure rates (failures are rarer than working states)
data['Sun'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])  # Sun usually present
data['PV Module'] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
data['Circuit Breaker'] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
data['Diode'] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
data['Fuse'] = np.random.choice([0, 1], size=n_samples, p=[0.96, 0.04])
data['Cable'] = np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])

# Add realistic noise – sometimes random simultaneous failures
for _ in range(5):
    idx = np.random.randint(0, n_samples)
    data.loc[idx, np.random.choice(components, 2, replace=False)] = 1

# Simulated "Power Supplied" measurement (1 = Fail to Supply = High Load)
def evaluate_high_load(row):
    # Define logic: high load (failure) if any of the following
    if row['Sun'] == 1:
        return 1
    if row['PV Module'] == 1:
        return 1
    if row['Circuit Breaker'] == 1 or row['Fuse'] == 1:
        return 1
    if row['Cable'] == 1:
        return 1
    return 0  # Otherwise, system likely supplies load

data['High Load'] = data.apply(evaluate_high_load, axis=1)

# # Save this as "realistic" log data
# data.to_csv("mock_real_world_data.csv", index=False)
# print("Mock real-world data saved as 'mock_real_world_data.csv'")

# Load mock real-world data
df = pd.read_csv("mock_real_world_data.csv")

# Drop any real-world specific columns if needed (timestamps, noise, etc.)
# In this case, data is already clean

# Convert to simplified truth table format
truth_table = df.copy()
truth_table.columns = [col.replace(" ", "_") for col in truth_table.columns]  # if needed for parsing

# truth_table.to_csv("simplified_truth_table.txt", sep=" ", index=False)
# print("Simplified state data saved as 'simplified_truth_table.txt'")


#%%

import numpy as np
import pandas as pd

#-------------------------------
# Step 1: Define Components & Probabilities
#-------------------------------
# Here, 0 means "working" and 1 means "failed".
# For the Sun, a failure (1) means insufficient sunlight.
components = {
    'Sun': 0.10,            # 10% chance of insufficient sunlight
    'PV Module': 0.05,      # 5% chance of PV module failure
    'Circuit Breaker': 0.02,
    'Diode': 0.03,
    'Fuse': 0.04,
    'Cable': 0.03
}

n_samples = 1000  # simulate 1000 time samples

#-------------------------------
# Step 2: Simulate Real World State Data
#-------------------------------
# For each component, randomly choose 0 or 1 based on its failure probability.
data = {comp: np.random.choice([0, 1], size=n_samples, p=[1 - p, p])
        for comp, p in components.items()}
df = pd.DataFrame(data)

#-------------------------------
# Step 3: Define the Top Event ("High Load")
#-------------------------------
# For our example, we assume that the system is considered to be under High Load (i.e. TE = 1)
# if any of these conditions is met:
#   - The Sun fails (insufficient sunlight)
#   - The PV Module fails
#   - Two or more of (Circuit Breaker, Diode, Fuse, Cable) fail
df['High Load'] = ((df['Sun'] == 1) | 
                   (df['PV Module'] == 1) | 
                   ((df['Circuit Breaker'] + df['Diode'] + df['Fuse'] + df['Cable']) >= 1)
                  ).astype(int)

#-------------------------------
# Step 4: Simplify the Data to a Truth Table
#-------------------------------
# In a real-world setting, you might have continuous logged data.
# Here we "simplify" it by extracting the unique rows (unique state combinations)
# so that we have one row per observed combination of component states and the top event.
truth_table_simulated = df.drop_duplicates().reset_index(drop=True)

# Optionally, sort the rows by the component states (for readability)
truth_table_simulated = truth_table_simulated.sort_values(by=list(components.keys())).reset_index(drop=True)

#-------------------------------
# Step 5: Export the Truth Table
#-------------------------------
# The final truth table includes columns for each component and the top event ("High Load").
# The file will be space-separated.
# truth_table_simulated.to_csv("pv_truth_table_simulated.txt", sep=" ", index=False)

print("Truth table generated and saved as 'pv_truth_table_simulated.txt'")
print(truth_table_simulated.head())
