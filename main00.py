import pandas as pd
from components.event import Event
from components.event import extract_event_number
from components.gate import Gate
from components.fault_tree import FaultTree
from components.fault_tree import print_fault_tree
from components.cutset import extract_cut_sets, get_minimal_cut_sets, build_boolean_expression
from components.boolean_parser import parse_expression, print_tree
from components.truth_table import generate_truth_table_from_expression
from components.validate import validate_truth_tables

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
