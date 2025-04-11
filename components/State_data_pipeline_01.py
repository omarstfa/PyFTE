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
