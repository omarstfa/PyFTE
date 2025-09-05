import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from components.event import Event
from components.event import extract_event_number
from components.gate import Gate
from components.fault_tree import FaultTree
from components.fault_tree import print_fault_tree
from components.cutset import extract_cut_sets, get_minimal_cut_sets, build_boolean_expression
from components.boolean_parser import parse_expression, print_tree
from components.truth_table import generate_truth_table_from_expression
from components.validate import validate_truth_tables
from components.critical_rank import calculate_importance_factors, simulate_reliability

# === Original Fault Tree Structure ===
topEvent = Event('Top Event')
or0 = Gate('OR', parent=topEvent)
intermediateEvent1 = Event('Intermediate Event 1', parent=or0)
intermediateEvent2 = Event('Intermediate Event 2', parent=or0)
or1 = Gate('OR', parent=intermediateEvent1)
Event('Basic Event 1', parent=or1)
Event('Basic Event 2', parent=or1)
intermediateEvent3 = Event('Intermediate Event 3', parent=or1)
or2 = Gate('OR', parent=intermediateEvent3)
intermediateEvent4 = Event('Intermediate Event 4', parent=or2)
intermediateEvent5 = Event('Intermediate Event 5', parent=or2)
or3 = Gate('OR', parent=intermediateEvent4)

for i in range(5, 17):
    Event(f'Basic Event {i}', parent=or3)

and1 = Gate('AND', parent=intermediateEvent5)
Event('Basic Event 3', parent=and1)
Event('Basic Event 4', parent=and1)

or4 = Gate('OR', parent=intermediateEvent2)
Event('Basic Event 17', parent=or4)
Event('Basic Event 18', parent=or4)

print("\nOriginal Fault Tree Structure:")
print_fault_tree(topEvent)

tree_original = FaultTree(topEvent)
truth_table_original = tree_original.generate_truth_table()
truth_table_original_simple = truth_table_original.copy()
column_mapping = {name: extract_event_number(name) for name in truth_table_original_simple.columns}
column_mapping["Top Event"] = "TE"
truth_table_original_simple.rename(columns=column_mapping, inplace=True)
sorted_cols = sorted([col for col in truth_table_original_simple.columns if col != "TE"], key=lambda x: int(x)) + ["TE"]
truth_table_original_simple = truth_table_original_simple[sorted_cols]
truth_table_original_simple.to_csv("truth_table_originalFT.txt", sep=' ', index=False)

print("\nOriginal Truth Table (sample):")
truth_table_original_BE = truth_table_original_simple.copy()
truth_table_original_BE.columns = [f"BE{col}" if col != 'TE' else 'TE' for col in truth_table_original_BE.columns]
print(truth_table_original_BE.head())

# === Cut Sets & Boolean Expression ===
cut_sets = extract_cut_sets(truth_table_original_simple)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)
boolean_expression = build_boolean_expression(minimal_cut_sets)

print("\nMinimal Cut Sets:")
for mcs in minimal_cut_sets:
    print(sorted(mcs))

print("\nExtracted Fault Tree Boolean Expression: TE =", boolean_expression.replace('.', '·'))

# === Construct Tree from Boolean Expression ===
print("\nConstructed Fault Tree Structure:")
root = parse_expression(boolean_expression)
print_tree(root)

# === Generated Truth Table from Boolean Expression ===
truth_table_constructed = generate_truth_table_from_expression(boolean_expression)
truth_table_constructed_BE = truth_table_constructed.copy()
truth_table_constructed_BE.columns = [f"BE{col}" if col != 'TE' else 'TE' for col in truth_table_constructed_BE.columns]
print("\nConstructed Truth Table (sample):")
print(truth_table_constructed_BE.head())

truth_table_constructed.to_csv("truth_table_constructedFT.txt", sep=" ", index=False)

# === Load and Validate Truth Tables ===
validate_truth_tables("truth_table_originalFT.txt", "truth_table_constructedFT.txt")


#%% Analysis: Importance & Unavailability

T_mission = 1000  # time units

# === Mapping and probabilities ===

label_map = {str(i): f'BE{i}' for i in range(1, 19)}

failure_rates = {
                'BE1':1e-7, 'BE2':2e-7,
                'BE3':8e-7, 'BE4':8e-7,
                'BE5':8.46e-6,  'BE6':4.9e-6,
                'BE7':3e-7, 'BE8':1.3e-6,
                'BE9':8.8e-6, 'BE10':1.115e-5,
                'BE11':1.487e-5,'BE12':1.01e-6,
                'BE13':2.1e-7,'BE14':5.2e-7,
                'BE15': 7.29e-6, 'BE16':5.7e-6,
                'BE17':1e-7,'BE18':2e-7
                }
repair_times = {
                'BE1': 3,  'BE2': 3,
                'BE3': 6,  'BE4': 6,
                'BE5': 12, 'BE6': 12,
                'BE7': 6,  'BE8': 6,
                'BE9': 6,  'BE10': 12,
                'BE11': 12,'BE12': 6,
                'BE13': 3, 'BE14': 3,
                'BE15': 12,'BE16': 12,
                'BE17': 3, 'BE18': 3
            }

# Convert set-style cut sets into lists of numbers (e.g., '1' -> 'BE1')
mcs_numeric = [[extract_event_number(be) for be in cut] for cut in minimal_cut_sets]

# === Compute Importance Factors ===

# Convert failure rates to failure probabilities for importance calc

failure_probs = {be: 1 - np.exp(-rate * T_mission) for be, rate in failure_rates.items()}

importance_df = calculate_importance_factors(mcs_numeric, label_map, failure_probs)
print("\n--- Importance Factors ---")
print(importance_df)

# === Simulate Reliability ===
minimal_cut_sets_BE = [[f'BE{extract_event_number(be)}' for be in cut] for cut in minimal_cut_sets]
sys_unavail, sys_ci, comp_df, availability_time_series, time_grid, all_component_states, availability_sys, MTTF_top, MTTR_top = simulate_reliability(minimal_cut_sets_BE, failure_rates, repair_times)


print("\nSystem Unavailability:", sys_unavail, "CI:", sys_ci, "System Availability:", availability_sys, "MTTF_Top:", MTTF_top, "MTTR_Top:", MTTR_top)
print("\nComponent Reliability Summary:\n", comp_df.sort_values('Unavailability', ascending=False))


#%% PLots


# Plot Unavailability
plt.figure(figsize=(10, 6))
ax = plt.gca()  # Get current Axes
ax.bar(comp_df.index, comp_df['Unavailability']/1000, label='Basic Events')
# ax.bar('Top Event', sys_unavail/1000, color='orange', label='Top Event')
ax.set_ylabel('Unavailability', fontsize=14)
ax.tick_params(axis='x', rotation=45, labelsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.yaxis.get_offset_text().set_fontsize(12)  # Set offset text font size
# ax.legend()
plt.tight_layout()
plt.show()


# Plot System Availability Over Time
plt.figure(figsize=(10, 6))
plt.plot(time_grid, availability_time_series, label='System Availability')
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Availability", fontsize=14)
# plt.title("System Availability Over Time", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
# plt.legend()
plt.show()


# # Plot Component Availability Over Time
# component_to_plot = 'BE10'
# comp_avail = 1 - np.mean(all_component_states[component_to_plot], axis=0)
# plt.figure(figsize=(10, 6))
# plt.plot(time_grid, comp_avail, label=f"{component_to_plot} Availability")
# plt.xlabel("Time (hours)", fontsize=14)
# plt.ylabel("Availability", fontsize=14)
# plt.title(f"{component_to_plot} Availability Over Time", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.show()


#%% Exporting Results Data

# Sort helper function
def sort_by_event_number(df):
    return df.sort_index(key=lambda x: x.str.extract(r'(\d+)').astype(int)[0])

# Sort importance and reliability DataFrames
importance_df_sorted = sort_by_event_number(importance_df)
comp_df_sorted = sort_by_event_number(comp_df)

comp_df_sorted.rename(columns={"Unavailability": "Unavailability (1e-3)"})
comp_df_sorted['Unavailability'] = comp_df_sorted['Unavailability']*1000


# # Export to CSV
# importance_df_sorted.to_csv("importance_factors_sorted.csv")
# comp_df_sorted.to_csv("reliability_metrics_sorted.csv")
# print("Exported sorted importance and reliability data to CSV.")
