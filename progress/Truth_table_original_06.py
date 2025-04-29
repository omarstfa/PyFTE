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
# Test3: 8 BE
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
# Test4: 10 BE
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
# Case Study
# =============================================================================

topEvent = Event('Top Event')

or0 = Gate('OR', parent=topEvent)
intermediateEvent1 = Event('Intermediate Event 1', parent=or0)
intermediateEvent2 = Event('Intermediate Event 2', parent=or0)

or1 = Gate('OR', parent=intermediateEvent1)
basicEvent1 = Event('Basic Event 1', parent=or1)
basicEvent2 = Event('Basic Event 2', parent=or1)
intermediateEvent3 = Event('Intermediate Event 3', parent=or1)

or2 = Gate('OR', parent=intermediateEvent3)
intermediateEvent4 = Event('Intermediate Event 4', parent=or2)
intermediateEvent5 = Event('Intermediate Event 5', parent=or2)

or3 = Gate('OR', parent=intermediateEvent4)
basicEvent5 = Event('Basic Event 5', parent=or3)
basicEvent6 = Event('Basic Event 6', parent=or3)
basicEvent7 = Event('Basic Event 7', parent=or3)
basicEvent8 = Event('Basic Event 8', parent=or3)
basicEvent9 = Event('Basic Event 9', parent=or3)
basicEvent10 = Event('Basic Event 10', parent=or3)
basicEvent11 = Event('Basic Event 11', parent=or3)
basicEvent12 = Event('Basic Event 12', parent=or3)
basicEvent13 = Event('Basic Event 13', parent=or3)
basicEvent14 = Event('Basic Event 14', parent=or3)
basicEvent15 = Event('Basic Event 15', parent=or3)
basicEvent16 = Event('Basic Event 16', parent=or3)

and1 = Gate('AND', parent=intermediateEvent5)
basicEvent3 = Event('Basic Event 3', parent=and1)
basicEvent4 = Event('Basic Event 4', parent=and1)

or4 = Gate('OR', parent=intermediateEvent2)
basicEvent16 = Event('Basic Event 17', parent=or4)
basicEvent17 = Event('Basic Event 18', parent=or4)

# =============================================================================

#%% Printing Original Fault Tree Strucuter

def print_fault_tree(node, prefix="", is_last=True, is_root=True):
    if is_root:
        print(f"{node.name}")
    else:
        connector = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        if isinstance(node, Event):
            print(f"{prefix}{connector}{node.name}")
        elif isinstance(node, Gate):
            print(f"{prefix}{connector}[{node.gate_type.upper()}]")
        prefix += next_prefix

    # Recurse into children
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        is_last_child = (i == child_count - 1)
        print_fault_tree(child, prefix, is_last_child, is_root=False)

print("\nOriginal Fault Tree Structure:")
print_fault_tree(topEvent)

#%% Build and Evaluate

import re

def extract_event_number(name):
    match = re.search(r'\d+', name)
    return match.group(0) if match else name

# Build and evaluate
tree_original = FaultTree(topEvent)
truth_table_original = tree_original.generate_truth_table()

truth_table_original_simple = truth_table_original.copy()

# Extract numeric labels from event names
column_mapping = {name: extract_event_number(name) for name in truth_table_original_simple.columns}
column_mapping["Top Event"] = "TE"
truth_table_original_simple.rename(columns=column_mapping, inplace=True)

# Sort columns numerically, keep TE last
sorted_cols = sorted([col for col in truth_table_original_simple.columns if col != "TE"], key=lambda x: int(x)) + ["TE"]
truth_table_original_simple = truth_table_original_simple[sorted_cols]

# Save to space-separated .txt
truth_table_original_simple.to_csv("truth_table_originalFT.txt", sep=' ', index=False, header=True)

print("\nOriginal Truth Table (sample):")
truth_table_original_BE= truth_table_original_simple.copy()
truth_table_original_BE.columns = [f"BE{col}" if col != 'TE' else 'TE' for col in truth_table_original_BE.columns]
print(truth_table_original_BE.head())

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

# Show a sample of the extracted cut sets
print("\n Identifying Cut Sets (first 5 shown):")
for i, cs in enumerate(cut_sets[:5]):  # Show only first 10 to avoid spamming
    print(f"Cut Set {i+1}: {[f"BE{e}" for e in sorted(cs)]}")

# Step 2: Find minimal cut sets
cut_sets = sorted(cut_sets, key=lambda x: len(x))  # Sort by size
minimal_cut_sets = []
for cs in cut_sets:
    if not any(cs.issuperset(mcs) for mcs in minimal_cut_sets):
        minimal_cut_sets.append(cs)

# Step 3: Create Boolean Expression
boolean_expression = " + ".join(["·".join(sorted(mcs)) for mcs in minimal_cut_sets])
boolean_expression_print = " + ".join(["·".join([f"BE{e}" for e in sorted(mcs)]) for mcs in minimal_cut_sets])

# Output
print("\nMinimal Cut Sets:")
for mcs in minimal_cut_sets:
    print(sorted(mcs))
print("\nExtracted Fault Tree Boolean Expression: TE =", boolean_expression_print)

#%% Boolean Algebra Factorization

# import pandas as pd
# from sympy import symbols, simplify_logic
# from sympy.parsing.sympy_parser import parse_expr

# # === Parse + Factor Boolean Expression using sympy ===

# def extract_variables(expr):
#     return sorted(set(re.findall(r'\b\d+\b', expr)), key=lambda x: int(x))

# def sympy_simplify_boolean(expr: str):
#     # Convert string like '2.5 + 1.5' into sympy form: 'x2 & x5 | x1 & x5'
#     vars_in_expr = extract_variables(expr)
#     sym_map = {v: symbols(f'x{v}') for v in vars_in_expr}
    
#     expr_sym = expr
#     for v in vars_in_expr:
#         expr_sym = re.sub(rf'\b{v}\b', f'x{v}', expr_sym)
    
#     expr_sym = expr_sym.replace('.', ' & ').replace('+', ' | ')
#     parsed = parse_expr(expr_sym, evaluate=False)
#     simplified = simplify_logic(parsed, form='dnf')  # or 'cnf'
    
#     return simplified, sym_map

# # === Convert simplified sympy expression to Fault Tree structure ===

# class Node:
#     def __init__(self, name, node_type="BASIC", children=None):
#         self.name = name
#         self.node_type = node_type  # BASIC, AND, OR
#         self.children = children if children else []

# def sympy_to_tree(expr, reverse_map):
#     if expr.func.__name__ == 'Symbol':
#         num = reverse_map[str(expr)]
#         return Node(num, node_type="BASIC")

#     node_type = 'OR' if expr.func.__name__ == 'Or' else 'AND'
#     children = [sympy_to_tree(arg, reverse_map) for arg in expr.args]
#     return Node(node_type, node_type=node_type, children=children)

# def print_tree(node, level=0):
#     indent = "  " * level
#     print(f"{indent}{node.name} ({node.node_type})")
#     for child in node.children:
#         print_tree(child, level + 1)


# # Step 1: Simplify
# simplified_expr, symbol_map = sympy_simplify_boolean(boolean_expression)

# # Step 2: Build reverse map (x1 → 1)
# reverse_map = {str(v): k for k, v in symbol_map.items()}

# # Step 3: Convert to tree
# root = Node("Top Event", node_type="OR", children=[sympy_to_tree(simplified_expr, reverse_map)])

# # Step 4: Visualize
# print("Simplified Boolean Expression:")
# print(simplified_expr)

# print("\n Fault Tree Structure:")
# print_tree(root)

#%% Build Fault Tree from Boolean Expression

class Node:
    def __init__(self, name, node_type="Basic Events", children=None):
        self.name = name
        self.node_type = node_type  # BASIC, AND, OR
        self.children = children if children else []
def parse_expression(expr: str) -> Node:
    """Parses a boolean expression into a fault tree structure."""
    terms = expr.split('+')
    terms = [term.strip() for term in terms]
    children = []

    for term in terms:
        factors = term.split('.')
        if len(factors) == 1:
            # Single basic event
            children.append(Node(factors[0], node_type="Basic Events"))
        else:
            # AND gate with multiple children
            and_children = [Node(f.strip(), node_type="Basic Events") for f in factors]
            and_node = Node(f"AND", node_type="AND", children=and_children)
            children.append(and_node)

    return Node("Top Event", node_type="OR", children=children)

def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node.name} ({node.node_type})")
    for child in node.children:
        print_tree(child, level + 1)


#%% Generating Truth Table from Boolean FT Expression


def extract_variables(expr):
    # Extract unique variable numbers using word boundaries
    return sorted(set(re.findall(r'\b\d+\b', expr)), key=lambda x: int(x))

def generate_truth_table_from_expression(expr: str):
    variables = extract_variables(expr)
    table = []

    for combo in itertools.product([0, 1], repeat=len(variables)):
        local_vars = {var: str(val) for var, val in zip(variables, combo)}
        
        # Replace each variable with its corresponding value using exact match
        expr_eval = expr
        for var, val in local_vars.items():
            expr_eval = re.sub(rf'\b{re.escape(var)}\b', val, expr_eval)

        # Convert expression to Python boolean logic syntax
        expr_eval = expr_eval.replace('+', ' or ').replace('·', ' and ')

        # Evaluate the expression
        result = eval(expr_eval)
        table.append([int(local_vars[var]) for var in variables] + [int(result)])

    df = pd.DataFrame(table, columns=variables + ['TE'])
    return df


# Part 1: Visual structure
print("\nConstructed Fault Tree Structure:")
root = parse_expression(boolean_expression)
print_tree(root)

# Part 2: Truth table
truth_table_constructed = generate_truth_table_from_expression(boolean_expression)
truth_table_constructed_BE= truth_table_constructed.copy()
truth_table_constructed_BE.columns = [f"BE{col}" if col != 'TE' else 'TE' for col in truth_table_constructed_BE.columns]
print("\n Constructed Truth Table (sample):")
print(truth_table_constructed_BE.head())

truth_table_constructed.to_csv("truth_table_from_constructed.txt", sep=" ", index=False)
print("\nTruth table generated and saved from Boolean expression.")


#%% Generating Truth Table from Boolean FT Expression

# truth_table_constructed = tree.generate_truth_table()

# # Optional: Rename and sort columns like before
# def extract_event_number(name): return re.search(r'\d+', name).group(0) if re.search(r'\d+', name) else name

# truth_table_constructed_simple = truth_table_constructed.copy()
# column_mapping = {name: extract_event_number(name) for name in truth_table_constructed_simple.columns}
# column_mapping["Top Event"] = "TE"
# truth_table_constructed_simple.rename(columns=column_mapping, inplace=True)
# sorted_cols = sorted([col for col in truth_table_constructed_simple.columns if col != "TE"], key=lambda x: int(x)) + ["TE"]
# truth_table_constructed_simple = truth_table_constructed_simple[sorted_cols]

# truth_table_constructed_simple.to_csv("truth_table_from_constructed.txt", sep=" ", index=False)
# print("\nTruth table generated and saved from Boolean expression.")

#%% Comparison of Original vs Constructed FT Truth Tables

# # Load the truth tables using space as the separator
# original = pd.read_csv("truth_table_originalFT.txt", sep=" ")
# constructed = pd.read_csv("truth_table_from_constructed.txt", sep=" ")

# # Ensure consistent column ordering, placing 'TE' (Top Event) last
# cols = sorted([col for col in constructed.columns if col != 'TE'], key=lambda x: int(x)) + ['TE']
# constructed, original = constructed[cols], original[cols]

# # Compare the Truth Tables
# if constructed.equals(original):
#     print("\n[Validation Successful]: Truth Tables Match!\n\nThe constructed fault tree produces an identical truth table to the original.")
# else:
#     print("\n[Validation Failed]: Truth Tables Do Not Match!\n\nDifferences detected between the constructed and original fault trees.")
#     diff = (constructed != original).any(axis=1)
#     mismatch = pd.concat([constructed[diff], original[diff]], axis=1, keys=["Constructed", "Original"])
#     print("\nFirst few mismatches:\n")
#     print(mismatch.head())
    
#%% Comparison of Original vs Constructed FT Truth Tables (ALL ROWS)

# =============================================================================
# Takes forever
# =============================================================================
# # Load the truth tables using space as the separator
# original = pd.read_csv("truth_table_originalFT.txt", sep=" ")
# constructed = pd.read_csv("truth_table_from_constructed.txt", sep=" ")
# 
# # Ensure consistent column ordering, placing 'TE' (Top Event) last
# cols = sorted([col for col in constructed.columns if col != 'TE'], key=lambda x: int(x)) + ['TE']
# constructed = constructed[cols]
# original = original[cols]
# 
# # Perform relaxed comparison
# unmatched_rows = []
# 
# for idx, row in constructed.iterrows():
#     # Check if this exact row (input combination + TE) exists in original
#     matches = (original == row).all(axis=1)
#     if not matches.any():
#         unmatched_rows.append(idx)
# 
# # Result
# if not unmatched_rows:
#     print("\n[Validation Successful]: Constructed truth table combinations all exist in original truth table!")
# else:
#     print(f"\n[Validation Warning]: {len(unmatched_rows)} unmatched rows found.")
#     mismatch = constructed.loc[unmatched_rows]
#     print("\nFirst few unmatched constructed rows:\n")
#     print(mismatch.head())
# =============================================================================


# Load the truth tables
original = pd.read_csv("truth_table_originalFT.txt", sep=" ")
constructed = pd.read_csv("truth_table_from_constructed.txt", sep=" ")

# Ensure consistent column ordering
cols = sorted([col for col in constructed.columns if col != 'TE'], key=lambda x: int(x)) + ['TE']
constructed = constructed[cols]
original = original[cols]

# Convert rows to sets of tuples for fast lookup
original_rows_set = set(tuple(row) for row in original.to_numpy())

# Find unmatched constructed rows
unmatched_indices = []

for idx, row in enumerate(constructed.to_numpy()):
    if tuple(row) not in original_rows_set:
        unmatched_indices.append(idx)

# Result
if not unmatched_indices:
    print("\n[Validation Successful]: Truth Tables Match!\n\nThe constructed fault tree produces an identical truth table to the original.")
else:
    print(f"\n[Validation Warning]: {len(unmatched_indices)} unmatched rows found.")
    mismatch = constructed.iloc[unmatched_indices]
    print("\nFirst few unmatched constructed rows:\n")
    print(mismatch.head())
