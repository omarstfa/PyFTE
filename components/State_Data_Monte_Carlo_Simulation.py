# Monte Carlo Simulation

import numpy as np
import pandas as pd

# Assume failure probabilities for each component (obtained from real system data)
failure_probs = {
    'Comp1': 0.05,
    'Comp2': 0.02,
    'Comp3': 0.03,
    'Comp4': 0.04,
    'Comp5': 0.01,
    'Comp6': 0.03
}

num_samples = 10000  # Number of simulation runs

def evaluate_fault_tree(state):
    # Replace with your real fault tree logic.
    if sum(state[comp] for comp in state) >= 1:
        return 1
    else:
        return 0

sim_data = []
for _ in range(num_samples):
    state = {comp: np.random.binomial(1, failure_probs[comp]) for comp in failure_probs}
    top_event = evaluate_fault_tree(state)
    sim_data.append([state[comp] for comp in failure_probs] + [top_event])

columns = list(failure_probs.keys()) + ['Top Event']
simulated_truth_table = pd.DataFrame(sim_data, columns=columns)
simulated_truth_table.to_csv("truth_table_simulated.txt", sep=" ", index=False)


#%% Enumerated Truth Table
import itertools
import pandas as pd

# Suppose we have six components and a function `evaluate_fault_tree(state)`
# that returns 1 if the top event occurs, 0 otherwise.
# `state` is a dictionary mapping component names to 0/1.

components = ['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6']

def evaluate_fault_tree(state):
    # Example logic: top event occurs if at least 3 components fail.
    # Replace with your actual fault tree logic.
    if sum(state[comp] for comp in components) >= 3:
        return 1
    else:
        return 0

# Generate all possible combinations
truth_table_data = []
for combo in itertools.product([0, 1], repeat=len(components)):
    state = dict(zip(components, combo))
    top_event = evaluate_fault_tree(state)
    truth_table_data.append(list(combo) + [top_event])

# Create DataFrame and name columns appropriately
columns = components + ['Top Event']
truth_table = pd.DataFrame(truth_table_data, columns=columns)
truth_table.to_csv("truth_table_enumerated.txt", sep=" ", index=False)
