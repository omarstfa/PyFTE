from components.event import Event
from components.gate import Gate
import itertools
import pandas as pd

class FaultTree:
    def __init__(self, top_event: Event):
        self.top_event = top_event
        self.all_events = []
        self.collect_events(top_event)

    def collect_events(self, node):
        if isinstance(node, Event):
            self.all_events.append(node)
        for child in node.children:
            self.collect_events(child)

    def get_basic_events(self):
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

def print_fault_tree(node, prefix="", is_last=True, is_root=True):
    if is_root:
        print(f"{node.name}")
    else:
        connector = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        if hasattr(node, 'name'):
            print(f"{prefix}{connector}{node.name}")
        prefix += next_prefix

    child_count = len(node.children)
    for i, child in enumerate(node.children):
        is_last_child = (i == child_count - 1)
        print_fault_tree(child, prefix, is_last_child, is_root=False)
