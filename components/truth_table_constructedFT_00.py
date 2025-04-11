# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 01:46:04 2025

@author: cj6253
"""

class Event:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.value = None
        if parent:
            parent.children.append(self)

    def evaluate(self):
        if self.children:
            return self.children[0].evaluate()
        return self.value

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
    def __init__(self, expression):
        self.basic_events = {}
        self.top_event = Event('Top Event')
        self.parse_expression(expression)

    def parse_expression(self, expr):
        or_gate = Gate('OR', parent=self.top_event)
        terms = expr.replace(' ', '').split('+')

        for term in terms:
            if '.' in term:
                and_gate = Gate('AND', parent=or_gate)
                for be_id in term.split('.'):
                    self.add_event(be_id, and_gate)
            else:
                self.add_event(term, or_gate)

    def add_event(self, be_id, parent):
        name = f"Basic Event {be_id}"
        event = self.basic_events.get(be_id)
        if not event:
            event = Event(name)
            self.basic_events[be_id] = event
        event.parent = parent
        parent.children.append(event)

    def get_basic_events(self):
        return [e for _, e in sorted(self.basic_events.items(), key=lambda x: int(x[0]))]

    def generate_truth_table(self):
        import itertools
        import pandas as pd

        basic_events = self.get_basic_events()
        columns = [str(i + 1) for i in range(len(basic_events))] + ['TE']
        table = []

        for combo in itertools.product([0, 1], repeat=len(basic_events)):
            for be, val in zip(basic_events, combo):
                be.value = val
            result = self.top_event.evaluate()
            table.append(list(combo) + [int(result)])

        df = pd.DataFrame(table, columns=columns)
        return df

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.top_event
        indent = '  ' * level
        label = f"{node.name} ({type(node).__name__})"
        print(f"{indent}{label}")
        for child in getattr(node, 'children', []):
            self.print_tree(child, level + 1)

#%% Truth Table 

expr = "1.2.3 + 4 + 5.6 + 7"

tree = FaultTree(expr)

print(" Fault Tree Structure:")
tree.print_tree()

print("\n Truth Table:")
df = tree.generate_truth_table()
print(df.head())

# Optional: Save to file
df.to_csv("truth_table_from_expression_test.txt", sep=' ', index=False)