# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 00:02:30 2025

@author: cj6253
"""

import itertools
import pandas as pd

class Event:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.value = None  # 0 or 1
        if parent:
            parent.children.append(self)

    def evaluate(self):
        # If this is an intermediate event with a gate child, evaluate gate
        if self.children:
            return self.children[0].evaluate()
        else:
            return self.value  # Basic event

class Gate:
    def __init__(self, gate_type, parent=None):
        self.gate_type = gate_type  # 'AND' or 'OR'
        self.children = []
        self.parent = parent
        if parent:
            parent.children.append(self)

    def evaluate(self):
        values = [child.evaluate() for child in self.children]
        return all(values) if self.gate_type == 'AND' else any(values)

class FaultTree:
    def __init__(self, top_event: Event):
        self.top_event = top_event
        self.all_events = []
        self.collect_events(top_event)

    def collect_events(self, node):
        """Recursively collect all event objects."""
        if isinstance(node, Event):
            self.all_events.append(node)
        for child in node.children:
            self.collect_events(child)

    def get_basic_events(self):
        """Return all leaf events (basic events with no children)."""
        return [event for event in self.all_events if not event.children]

    def generate_truth_table(self):
        basic_events = self.get_basic_events()
        columns = [be.name for be in basic_events] + ["Top Event"]
        table = []

        for combo in itertools.product([0, 1], repeat=len(basic_events)):
            for be, val in zip(basic_events, combo):
                be.value = val
            top_value = self.top_event.evaluate()
            table.append(list(combo) + [int(top_value)])

        return pd.DataFrame(table, columns=columns)



#%% Create fault tree structure

# =============================================================================
# Test0: Only ORs
# =============================================================================
# topEvent = Event('Top Event')

# or1 = Gate('OR', parent=topEvent)
# intermediateEvent1 = Event('Intermediate Event 1', parent=or1)
# basicEvent1 = Event('Basic Event 1', parent=or1)

# or2 = Gate('OR', parent=intermediateEvent1)
# basicEvent2 = Event('Basic Event 2', parent=or2)
# basicEvent3 = Event('Basic Event 3', parent=or2)
# =============================================================================

# =============================================================================
# Test1: Only ANDs
# =============================================================================
# topEvent = Event('Top Event')

# and1 = Gate('AND', parent=topEvent)
# intermediateEvent1 = Event('Intermediate Event 1', parent=and1)
# basicEvent1 = Event('Basic Event 1', parent=and1)

# and2 = Gate('AND', parent=intermediateEvent1)
# basicEvent2 = Event('Basic Event 2', parent=and2)
# basicEvent3 = Event('Basic Event 3', parent=and2)
# =============================================================================

# =============================================================================
# Test2: OR and AND
# =============================================================================
# topEvent = Event('Top Event')

# or1 = Gate('OR', parent=topEvent)
# intermediateEvent1 = Event('Intermediate Event 1', parent=or1)
# basicEvent1 = Event('Basic Event 1', parent=or1)

# and1 = Gate('AND', parent=intermediateEvent1)
# basicEvent2 = Event('Basic Event 2', parent=and1)
# basicEvent3 = Event('Basic Event 3', parent=and1)
# =============================================================================


# =============================================================================
# 8 BE
# =============================================================================

# topEvent = Event('Top Event')
# and1 = Gate('AND', parent=topEvent)
# intermediateEvent1 = Event('Intermediate Event 1', parent=and1)
# intermediateEvent2 = Event('Intermediate Event 2', parent=and1)
# or1 = Gate('OR', parent=intermediateEvent1)
# basicEvent1 = Event('Basic Event 1', parent=or1)
# basicEvent2 = Event('Basic Event 2', parent=or1)
# intermediateEvent3 = Event('Intermediate Event 3', parent=or1)
# and2 = Gate('AND', parent=intermediateEvent3)
# basicEvent3 = Event('Basic Event 3', parent=and2)
# basicEvent4 = Event('Basic Event 4', parent=and2)
# or2 = Gate('OR', parent=intermediateEvent2)
# basicEvent5 = Event('Basic Event 5', parent=or2)
# intermediateEvent4 = Event('Intermediate Event 4', parent=or2)
# and3 = Gate('AND', parent=intermediateEvent4)
# basicEvent6 = Event('Basic Event 6', parent=and3)
# basicEvent7 = Event('Basic Event 7', parent=and3)
# basicEvent8 = Event('Basic Event 8', parent=and3) 

# =============================================================================

# =============================================================================
# 10 BE
# =============================================================================

# topEvent = Event('Top Event')

# and1 = Gate('AND', parent=topEvent)
# intermediateEvent1 = Event('Intermediate Event 1', parent=and1)
# intermediateEvent2 = Event('Intermediate Event 2', parent=and1)

# or1 = Gate('OR', parent=intermediateEvent1)
# basicEvent1 = Event('Basic Event 1', parent=or1)
# basicEvent2 = Event('Basic Event 2', parent=or1)
# intermediateEvent3 = Event('Intermediate Event 3', parent=or1)

# and2 = Gate('AND', parent=intermediateEvent3)
# basicEvent3 = Event('Basic Event 3', parent=and2)
# basicEvent4 = Event('Basic Event 4', parent=and2)

# or2 = Gate('OR', parent=intermediateEvent2)
# basicEvent5 = Event('Basic Event 5', parent=or2)
# intermediateEvent4 = Event('Intermediate Event 4', parent=or2)

# and3 = Gate('AND', parent=intermediateEvent4)
# basicEvent6 = Event('Basic Event 6', parent=and3)
# basicEvent7 = Event('Basic Event 7', parent=and3)
# basicEvent8 = Event('Basic Event 8', parent=and3)
# intermediateEvent5 = Event('Intermediate Event 5', parent=and3)

# or3 = Gate('OR', parent=intermediateEvent5)
# basicEvent9 = Event('Basic Event 9', parent=or3)
# basicEvent10 = Event('Basic Event 10', parent=or3)

# =============================================================================

# =============================================================================
# PV System
# =============================================================================

topEvent = Event('Top Event')

or0 = Gate('or', parent=topEvent)
intermediateEvent1 = Event('Intermediate Event 1', parent=or0)
intermediateEvent2 = Event('Intermediate Event 2', parent=or0)

or1 = Gate('OR', parent=intermediateEvent1)
basicEvent1 = Event('Basic Event 1', parent=or1)
basicEvent2 = Event('Basic Event 2', parent=or1)
intermediateEvent3 = Event('Intermediate Event 3', parent=or1)

or3 = Gate('OR', parent=intermediateEvent3)
basicEvent3 = Event('Basic Event 3', parent=or3)
basicEvent4 = Event('Basic Event 4', parent=or3)
intermediateEvent4 = Event('Intermediate Event 4', parent=or3)

or4 = Gate('OR', parent=intermediateEvent4)
basicEvent5 = Event('Basic Event 5', parent=or4)
basicEvent6 = Event('Basic Event 6', parent=or4)
basicEvent7 = Event('Basic Event 7', parent=or4)
basicEvent8 = Event('Basic Event 8', parent=or4)
basicEvent9 = Event('Basic Event 9', parent=or4)
basicEvent10 = Event('Basic Event 10', parent=or4)
basicEvent11 = Event('Basic Event 11', parent=or4)

or5 = Gate('OR', parent=intermediateEvent2)
basicEvent12 = Event('Basic Event 12', parent=or5)
basicEvent13 = Event('Basic Event 13', parent=or5)

# =============================================================================



#%% Build and Evaluate

import re

def extract_event_number(name):
    match = re.search(r'\d+', name)
    return match.group(0) if match else name

# Build and evaluate
tree = FaultTree(topEvent)
truth_table = tree.generate_truth_table()

truth_table_simple = truth_table.copy()

# Extract numeric labels from event names
column_mapping = {name: extract_event_number(name) for name in truth_table.columns}
column_mapping["Top Event"] = "TE"
truth_table_simple.rename(columns=column_mapping, inplace=True)

# Sort columns numerically, keep TE last
sorted_cols = sorted([col for col in truth_table_simple.columns if col != "TE"], key=lambda x: int(x)) + ["TE"]
truth_table_simple = truth_table_simple[sorted_cols]

# Save to space-separated .txt
truth_table_simple.to_csv("truth_table_originalFT.txt", sep=' ', index=False, header=True)


#%% Re-constructing FT Boolean Expression

# Load the truth table
filename = 'truth_table_originalFT.txt'
data = pd.read_csv(filename, sep=r'\s+', header=0)

# Separate basic events and top event
basic_events = data.columns[:-1]
TE = data.columns[-1]

# Step 1: Identify all cut sets (rows where TE == 1)
cut_sets = []
for _, row in data.iterrows():
    if row[TE] == 1:
        cut_sets.append(set(basic_events[row[basic_events] == 1]))

# Step 2: Find minimal cut sets
cut_sets = sorted(cut_sets, key=lambda x: len(x))  # Sort by size
minimal_cut_sets = []
for cs in cut_sets:
    if not any(cs.issuperset(mcs) for mcs in minimal_cut_sets):
        minimal_cut_sets.append(cs)

# Step 3: Create Boolean Expression
boolean_expression = " + ".join([".".join(sorted(mcs)) for mcs in minimal_cut_sets])

# Output
print("Minimal Cut Sets:")
for mcs in minimal_cut_sets:
    print(sorted(mcs))
print("\nBoolean Fault Tree Expression: TE =", boolean_expression)


#%% Generating Truth Table from Boolean FT Expression

# def build_tree_from_boolean(expression: str):
#     from collections import defaultdict

#     topEvent = Event("Top Event")
#     or_gate = Gate("OR", parent=topEvent)

#     terms = expression.replace("TE =", "").strip().split("+")
#     terms = [term.strip() for term in terms]

#     used_event_names = set()
#     basic_event_objects = {}

#     for i, term in enumerate(terms):
#         factors = term.split(".")
#         factors = [f.strip() for f in factors]

#         # Handle multi-event AND gate
#         if len(factors) > 1:
#             and_gate = Gate("AND", parent=or_gate)
#             for be_num in factors:
#                 event_name = f"Basic Event {be_num}"
#                 if event_name not in basic_event_objects:
#                     basic_event_objects[event_name] = Event(event_name, parent=and_gate)
#                 else:
#                     Event(event_name, parent=and_gate)  # reuse name, create new leaf
#         else:
#             be_num = factors[0]
#             event_name = f"Basic Event {be_num}"
#             if event_name not in basic_event_objects:
#                 basic_event_objects[event_name] = Event(event_name, parent=or_gate)
#             else:
#                 Event(event_name, parent=or_gate)

#     return FaultTree(topEvent)

# tree = build_tree_from_boolean(boolean_expression)


truth_table = tree.generate_truth_table()

# Optional: Rename and sort columns like before
def extract_event_number(name): return re.search(r'\d+', name).group(0) if re.search(r'\d+', name) else name

truth_table_simple = truth_table.copy()
column_mapping = {name: extract_event_number(name) for name in truth_table.columns}
column_mapping["Top Event"] = "TE"
truth_table_simple.rename(columns=column_mapping, inplace=True)
sorted_cols = sorted([col for col in truth_table_simple.columns if col != "TE"], key=lambda x: int(x)) + ["TE"]
truth_table_simple = truth_table_simple[sorted_cols]

truth_table_simple.to_csv("truth_table_from_expression.txt", sep=" ", index=False)
print("Truth table generated and saved from Boolean expression.")

#%% Comparison of Original vs Constructed FT Truth Tables