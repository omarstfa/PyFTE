from components.event import Event
from components.event import extract_event_number
from components.gate import Gate
from components.fault_tree import FaultTree
from components.fault_tree import print_fault_tree
from components.cutset import extract_cut_sets, get_minimal_cut_sets, build_boolean_expression
from components.boolean_parser import parse_expression, print_tree
from components.truth_table import generate_truth_table_from_expression

import pandas as pd

# Build fault tree
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

# Print original fault tree
print("\nOriginal Fault Tree Structure:")
print_fault_tree(topEvent)

# Generate truth table
tree = FaultTree(topEvent)
df = tree.generate_truth_table()
df_simple = df.rename(columns={col: extract_event_number(col) for col in df.columns})
df_simple = df_simple[[c for c in sorted(df_simple.columns, key=lambda x: (x != "TE", int(extract_event_number(x))))]]
df_simple.to_csv("truth_table_originalFT.txt", sep=" ", index=False)

# Extract cut sets and Boolean expression
cut_sets = extract_cut_sets(df_simple)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)
expr = build_boolean_expression(minimal_cut_sets)
print("\nExtracted Fault Tree Boolean Expression:\nTE =", expr)

# Construct fault tree from Boolean expression
print("\nConstructed Fault Tree Structure:")
ft_root = parse_expression(expr)
print_tree(ft_root)

# Generate truth table from expression
df_expr = generate_truth_table_from_expression(expr)
df_expr.to_csv("truth_table_from_constructed.txt", sep=" ", index=False)

# Compare truth tables
original = pd.read_csv("truth_table_originalFT.txt", sep=" ")
constructed = pd.read_csv("truth_table_from_constructed.txt", sep=" ")
cols = sorted([col for col in constructed.columns if col != 'TE'], key=lambda x: int(x)) + ['TE']
constructed = constructed[cols]
original = original[cols]
if constructed.equals(original):
    print("\n[Validation Successful]: Truth Tables Match!")
else:
    print("\n[Validation Failed]: Truth Tables Do Not Match!")
