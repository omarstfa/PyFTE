# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:17:20 2025

@author: cj6253
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Input data
failure_rates = [
                    1e-6,  # BE1
                    2e-6,  # BE2
                    8e-6,  # BE3
                    8e-6,  # BE4
                    8.46e-5,  # BE5
                    4.9e-5,   # BE6
                    3e-6,     # BE7
                    1.3e-5,   # BE8
                    8.8e-5,   # BE9
                    1.115e-4, # BE10
                    1.487e-4, # BE11
                    1.01e-5,  # BE12
                    2.1e-6,   # BE13
                    5.2e-6,   # BE14
                    7.29e-5,  # BE15
                    5.7e-5,   # BE16
                    1e-6,     # BE17
                    2e-6      # BE18
                ]

# Assign to basic_events dictionary
basic_events = {
    f'BE{i+1}': {'fail_prob': failure_rates[i], 'repair_mean': 50}
    for i in range(18)
}
minimal_cut_sets = [
    ['BE18'], ['BE17'], ['BE16'], ['BE15'], ['BE14'], ['BE13'], ['BE12'],
    ['BE11'], ['BE10'], ['BE9'], ['BE8'], ['BE7'], ['BE6'], ['BE5'],
    ['BE2'], ['BE1'], ['BE3','BE4']
]

#%% Simulation parameters
T = 1000         # mission time
dt = 1           # time step
time_grid = np.arange(0, T+dt, dt)
N_SIM = 5000     # Monte Carlo runs

# Simulate each component: repair and failure events
def simulate_component(f_rate, repair_mean):
    t = 0
    events = []
    while t < T:
        t += np.random.exponential(1/f_rate)
        if t >= T:
            break
        t_down = t
        t += np.random.exponential(repair_mean)
        events.append((t_down, min(t, T)))
    return events

#%% Monte Carlo simulation
sys_avail = np.zeros(N_SIM)
component_stats = {be: {'up_times': [], 'down_times': []} for be in basic_events}
top_event_failures = []
top_event_down_times = []
top_event_states = np.zeros((N_SIM, len(time_grid)), dtype=bool)

# Track all component states across all simulations
all_component_states = {
    be: np.zeros((N_SIM, len(time_grid)), dtype=bool) for be in basic_events
}

for sim in range(N_SIM):
    # Track downtime arrays for each component
    comp_states = {be: np.zeros_like(time_grid, dtype=bool) for be in basic_events}
    for be, params in basic_events.items():
        f_rate = params['fail_prob']
        events = simulate_component(f_rate, params['repair_mean'])
        for down, up in events:
            idx = (time_grid >= down) & (time_grid < up)
            comp_states[be][idx] = True
            all_component_states[be][sim][idx] = True
            component_stats[be]['down_times'].append(up - down)

    # Top event state (system failure state)
    top_state = np.zeros_like(time_grid, dtype=bool)
    for cut in minimal_cut_sets:
        mask = np.logical_and.reduce([comp_states[e] for e in cut])
        top_state |= mask

    # Record top event availability for this sim
    top_event_states[sim] = top_state
    sys_avail[sim] = 1 - np.mean(top_state)

    # Record first failure time (MTTF)
    fail_indices = np.where(top_state)[0]
    if len(fail_indices) > 0:
        first_failure_time = time_grid[fail_indices[0]]
    else:
        first_failure_time = T
    top_event_failures.append(first_failure_time)

    # Record system down durations (MTTR)
    down = False
    start = None
    for i, state in enumerate(top_state):
        if state and not down:
            down = True
            start = time_grid[i]
        elif not state and down:
            down = False
            top_event_down_times.append(time_grid[i] - start)
    if down:
        top_event_down_times.append(T - start)

    # Component availability
    for be in basic_events:
        component_stats[be]['up_times'].append(np.mean(~comp_states[be]))

#%% Aggregation
comp_df = pd.DataFrame({
    be: {
        'Failure Rate': -np.log(1 - basic_events[be]['fail_prob']),
        'Unavailability': 1 - np.mean(stats['up_times']),
        'MTBF': np.mean(stats['up_times']) * T / len(stats['down_times']) if stats['down_times'] else np.nan,
        'MTTR': np.mean(stats['down_times']) if stats['down_times'] else np.nan,
    }
    for be, stats in component_stats.items()
}).T

sys_unavail = 1 - np.mean(sys_avail)
sys_ci = np.percentile(sys_avail, [2.5, 97.5])
MTTF_top = np.mean(top_event_failures)
MTTR_top = np.mean(top_event_down_times)
availability_top = MTTF_top / (MTTF_top + MTTR_top)

print("System Unavailability:", sys_unavail, "CI:", sys_ci)
print("Top Event MTTF:", MTTF_top)
print("Top Event MTTR:", MTTR_top)
print("Top Event Availability (from MTTF/MTTR):", availability_top)
print("\nComponent Summary:\n", comp_df.sort_values('Unavailability', ascending=False))

#%% Plot: Bar chart of component/system unavailability
plt.figure(figsize=(10,6))
for be in comp_df.index:
    plt.bar(be, comp_df.loc[be, 'Unavailability'], color='gray')
plt.bar('Top Event', sys_unavail, color='red')
plt.ylabel('Unavailability')
plt.title('Unavailability of Components & System over simulations')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#%% Plot: System availability over time (cut at first near-zero point)
system_availability_time = 1 - np.mean(top_event_states, axis=0)

# Trim time and availability arrays
system_availability_trim = system_availability_time[:40000]
time_trimmed = time_grid[:40000]

plt.figure(figsize=(10, 6))
plt.plot(time_trimmed, system_availability_trim, label='System Availability')
plt.xlabel("Time")
plt.ylabel("System Availability")
plt.title("Top Event (System) Availability")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

#%% Plot: Availability of a specific component until it reaches zero
component_to_plot = 'BE5'  # Change this as needed

# Availability over time = 1 - mean down state
component_avail_time = 1 - np.mean(all_component_states[component_to_plot], axis=0)

# Trim time and availability arrays
component_avail_trim = component_avail_time[:1000]
time_trimmed = time_grid[:1000]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_trimmed, component_avail_trim, label=f'{component_to_plot} Availability')
plt.xlabel("Time")
plt.ylabel("Availability")
plt.title(f"{component_to_plot} Availability")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
