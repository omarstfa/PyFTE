import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Step 1: Define Basic Events and Their Failure Probabilities
failure_probs = {
    'Fuse_Oxidation': 0.0001,
    'Fuse_Improper_Maintenance': 0.0002,
    'MCB_Faulty': 0.0008,
    'PV_Broken_Interconnect': 0.0846,
    'PV_Grounding_System_Fault': 0.0490,
    'PV_Glass_Breakage': 0.0003,
    'PV_Soiling': 0.0013,
    'PV_Shading': 0.0088,
    'PV_Broken_Cell': 0.1115,
    'PV_Solder_Bond_Failure': 0.1487,
    'PV_Hot_Spot': 0.0101,
    'PV_Faulty_Bypass_Diode': 0.0021,
    'PV_Short_Open_Circuit': 0.0052,
    'Rack_Structure_Fault': 0.0729,
    'Encapsulant_Fault': 0.0570,
    'Cable_Insulation_Failure': 0.0001,
    'Cable_Material_Aging': 0.0002
}

# Severities based on common sense
severity_mapping = {
    'Fuse_Oxidation': 'Medium',
    'Fuse_Improper_Maintenance': 'Medium',
    'MCB_Faulty': 'High',
    'PV_Broken_Interconnect': 'High',
    'PV_Grounding_System_Fault': 'High',
    'PV_Glass_Breakage': 'Medium',
    'PV_Soiling': 'Low',
    'PV_Shading': 'Low',
    'PV_Broken_Cell': 'High',
    'PV_Solder_Bond_Failure': 'High',
    'PV_Hot_Spot': 'Medium',
    'PV_Faulty_Bypass_Diode': 'Medium',
    'PV_Short_Open_Circuit': 'Medium',
    'Rack_Structure_Fault': 'Medium',
    'Encapsulant_Fault': 'Medium',
    'Cable_Insulation_Failure': 'Medium',
    'Cable_Material_Aging': 'Medium'
}

# Generate simulated data
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)
days = (end_date - start_date).days

num_events = 1000  # Number of fault events to simulate

log_entries = []
for _ in range(num_events):
    component = random.choice(list(failure_probs.keys()))
    probability = failure_probs[component]
    if np.random.rand() < probability * 100:  # scaled up since probabilities are very low
        # Random timestamp within the year
        random_days = random.randint(0, days)
        random_seconds = random.randint(0, 24*3600)
        fault_time = start_date + timedelta(days=random_days, seconds=random_seconds)

        # Log active fault
        entry_active = {
            'Timestamp': fault_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Component': component,
            'Description': f"{component.replace('_', ' ')} fault detected.",
            'Fault_Code': component.split('_')[0][:2].upper() + '_' + str(random.randint(100,999)),
            'Severity': severity_mapping.get(component, 'Medium'),
            'Status': 'Active',
            'Value': f"SimValue_{np.random.randint(10, 100)}"
        }
        log_entries.append(entry_active)

        # Random clearing time (within 1-12 hours after fault)
        clear_time = fault_time + timedelta(hours=random.randint(1, 12))
        entry_cleared = entry_active.copy()
        entry_cleared['Timestamp'] = clear_time.strftime('%Y-%m-%d %H:%M:%S')
        entry_cleared['Status'] = 'Cleared'
        log_entries.append(entry_cleared)

# Convert to DataFrame
fault_log_df = pd.DataFrame(log_entries)

# Save full fault log
fault_log_df = fault_log_df.sort_values('Timestamp').reset_index(drop=True)
fault_log_df.to_csv("synthetic_fault_log_realistic.csv", index=False)

print("Synthetic realistic PV system fault log generated!")
print(fault_log_df.head())

# Extract Truth Table from Fault Log

# Load the fault log
df_faults = pd.read_csv("synthetic_fault_log_realistic.csv")

# Convert timestamps to datetime
df_faults['Timestamp'] = pd.to_datetime(df_faults['Timestamp'])

# Create a timeline from the minimum to maximum timestamp at a fixed interval (e.g., every 1 hour)
timeline = pd.date_range(df_faults['Timestamp'].min(), df_faults['Timestamp'].max(), freq='1H')

# Initialize states: assume all components are healthy initially
components = sorted(df_faults['Component'].unique())
current_state = {comp: 0 for comp in components}

truth_table = []

# Create event queue
events = df_faults.sort_values('Timestamp').to_dict('records')
event_idx = 0

for time_point in timeline:
    # Update states based on events up to the current time
    while event_idx < len(events) and events[event_idx]['Timestamp'] <= time_point:
        event = events[event_idx]
        component = event['Component']
        status = event['Status']
        current_state[component] = 1 if status == 'Active' else 0
        event_idx += 1

    # Snapshot state at this time
    snapshot = [current_state[comp] for comp in components]
    TE = 1 if any(snapshot) else 0
    truth_table.append(snapshot + [TE])

# Create DataFrame
truth_table_df = pd.DataFrame(truth_table, columns=components + ['TE'])

# Save truth table
truth_table_df.to_csv("extracted_truth_table_from_fault_log.csv", index=False)

print("Truth table extracted from fault log!")
print(truth_table_df.head())
