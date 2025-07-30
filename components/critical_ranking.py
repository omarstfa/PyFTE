# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:17:20 2025

@author: cj6253
"""
import pandas as pd

#%% Input data
mcs = [
        ['18'], ['17'], ['16'], ['15'], ['14'], ['13'], ['12'],
        ['11'], ['10'], ['9'], ['8'], ['7'], ['6'], ['5'],
        ['2'], ['1'], ['3','4']
        ]


label_map = {str(i): f'BE{i}' for i in range(1, 19)}
failure_probs = {
                'BE1':0.0001, 'BE2':0.0002, 'BE3':0.0008, 'BE4':0.0008,
                'BE5':0.0846,  'BE6':0.049, 'BE7':0.0003, 'BE8':0.0013,
                'BE9':0.0088, 'BE10':0.1115,'BE11':0.1487,'BE12':0.0101,
                'BE13':0.0021,'BE14':0.0052,'BE15': 0.0729, 'BE16':0.057,
                'BE17':0.0001,'BE18':0.0002
                }

#%% Calculating Importance Factors

def prod(iterable):
    p = 1
    for x in iterable:
        p *= x
    return p

def top_prob(probs):
    return sum(prod(probs[label_map[e]] for e in cut) for cut in mcs)

# Top event base probability
Q0 = top_prob(failure_probs)

# Compute structure importance
structure_importance = {}
for ev in label_map:
    terms = []
    for cut in mcs:
        if ev in cut:
            Nj = len(cut)
            terms.append(1 - 1 / (2 ** (Nj - 1)))
    I_phi = 1 - prod(terms) if terms else 0
    structure_importance[label_map[ev]] = I_phi

# Now calculate other importances
results = []
for ev in label_map:
    code = label_map[ev]
    
    probs1 = failure_probs.copy()
    probs0 = failure_probs.copy()
    probs1[code] = 1.0
    probs0[code] = 0.0
    
    Q1 = top_prob(probs1)
    Q0i = top_prob(probs0)
    
    IB = Q1 - Q0i  # Birnbaum
    qi = failure_probs[code]
    ICR = IB * qi / Q0 if Q0 > 0 else None  # Criticality
    
    FV = sum(
        prod(failure_probs[label_map[e]] for e in cut)
        for cut in mcs if ev in cut
    )
    IFV = FV / Q0 if Q0 > 0 else None

    results.append({
        'Event': code,
        'Structure': structure_importance[code],
        'Birnbaum': IB,
        'Criticality': ICR,
        'Fussell-Vesely': IFV
    })

df = pd.DataFrame(results).set_index('Event')
df = df.sort_values(by='Criticality', ascending=False)
print(df)

#%% Reliability

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input data
basic_events = {f'BE{i}': {'fail_prob': p, 'repair_mean': 50}  # mean repair time 50 units
                for i, p in zip(range(1,19),
                                [0.0001,0.0002,0.0008,0.0008,0.0846,0.049,0.0003,0.0013,
                                 0.0088,0.1115,0.1487,0.0101,0.0021,0.0052,
                                 0.0729,0.057,0.0001,0.0002])}

minimal_cut_sets = [
    ['BE18'], ['BE17'], ['BE16'], ['BE15'], ['BE14'], ['BE13'], ['BE12'],
    ['BE11'], ['BE10'], ['BE9'], ['BE8'], ['BE7'], ['BE6'], ['BE5'],
    ['BE2'], ['BE1'], ['BE3','BE4']
]

# Simulation parameters
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
        if t >= T: break
        t_down = t
        t += np.random.exponential(repair_mean)
        events.append((t_down, min(t, T)))
    return events

# Monte Carlo simulation
sys_avail = np.zeros(N_SIM)
component_stats = {be: {'up_times': [], 'down_times': []} for be in basic_events}

for sim in range(N_SIM):
    # Track downtime arrays for each component
    comp_states = {be: np.zeros_like(time_grid, dtype=bool) for be in basic_events}
    for be, params in basic_events.items():
        f_rate = -np.log(1-params['fail_prob'])
        events = simulate_component(f_rate, params['repair_mean'])
        for down, up in events:
            idx = (time_grid >= down) & (time_grid < up)
            comp_states[be][idx] = True
            component_stats[be]['down_times'].append(up-down)
    # Top event state
    top_state = np.zeros_like(time_grid, dtype=bool)
    for cut in minimal_cut_sets:
        mask = np.logical_and.reduce([comp_states[e] for e in cut])
        top_state |= mask

    sys_avail[sim] = 1 - np.mean(top_state)
    for be in basic_events:
        component_stats[be]['up_times'].append(np.mean(~comp_states[be]))

# Aggregate results
comp_df = pd.DataFrame({
    be: {
        'Failure Rate': -np.log(1 - failure_probs[be]),
        'Unavailability': 1 - np.mean(stats['up_times']),
        'MTBF': np.mean(stats['up_times']) * T / len(stats['down_times']) if stats['down_times'] else np.nan,
        'MTTR': np.mean(stats['down_times']) if stats['down_times'] else np.nan,
    }
    for be, stats in component_stats.items()
}).T

sys_unavail = 1 - np.mean(sys_avail)
sys_ci = np.percentile(sys_avail, [2.5, 97.5])

print("System Unavailability:", sys_unavail, "CI:", sys_ci)
print("\nComponent Summary:\n", comp_df.sort_values('Unavailability', ascending=False))

# --- Plotting
plt.figure(figsize=(10,6))
for be in comp_df.index:
    plt.bar(be, comp_df.loc[be, 'Unavailability'], color='gray')
plt.bar('Top Event', sys_unavail, color='red')
plt.ylabel('Unavailability')
plt.title('Unavailability of Components & System over simulations')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
