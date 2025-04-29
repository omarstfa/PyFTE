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
        self.value = None  # Used for evaluation
        if parent:
            parent.children.append(self)

class Gate:
    def __init__(self, gate_type, parent=None):
        self.gate_type = gate_type  # 'AND' or 'OR'
        self.children = []
        self.parent = parent
        if parent:
            parent.children.append(self)
        self.value = None

    def evaluate(self):
        values = [child.evaluate() if isinstance(child, (Gate, Event)) else child for child in self.children]
        if self.gate_type == 'AND':
            return all(values)
        elif self.gate_type == 'OR':
            return any(values)

# Override evaluate for Event
def event_evaluate(self):
    if self.children:
        return self.children[0].evaluate()
    else:
        return self.value
Event.evaluate = event_evaluate

# Create fault tree structure

# =============================================================================
# Only ORs
# =============================================================================
topEvent = Event('Top Event')

or1 = Gate('OR', parent=topEvent)
intermediateEvent1 = Event('Intermediate Event 1', parent=or1)
basicEvent1 = Event('Basic Event 1', parent=or1)

or2 = Gate('OR', parent=intermediateEvent1)
basicEvent2 = Event('Basic Event 2', parent=or2)
basicEvent3 = Event('Basic Event 3', parent=or2)
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

# intermediateEvent5 = Event('Intermediate Event 5', parent=intermediateEvent2)
# or3 = Gate('OR', parent=intermediateEvent5)
# basicEvent9 = Event('Basic Event 9', parent=or3)
# basicEvent10 = Event('Basic Event 10', parent=or3)

# List of all basic events
basic_events = [
    basicEvent1, basicEvent2, basicEvent3,
    # basicEvent4, basicEvent5,
    # basicEvent6, basicEvent7, basicEvent8, basicEvent9, basicEvent10
]

# Generate truth table
columns = [be.name for be in basic_events] + ["Top Event"]
table = []

for combo in itertools.product([0, 1], repeat=len(basic_events)):
    # Assign values to basic events
    for be, val in zip(basic_events, combo):
        be.value = val

    # Evaluate top event
    top_value = topEvent.evaluate()
    table.append(list(combo) + [int(top_value)])

# Create DataFrame and save to CSV
df = pd.DataFrame(table, columns=columns)
df.to_csv("generated_truth_table.txt", sep="\t", index=False)

print("Generated truth table with", len(df), "rows.")
