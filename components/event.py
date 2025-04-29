import re

class Event:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.value = None  # 0 or 1
        if parent:
            parent.children.append(self)

    def evaluate(self):
        if self.children:
            return self.children[0].evaluate()
        else:
            return self.value  # Basic event

def extract_event_number(name):
    match = re.search(r'\d+', name)
    return match.group(0) if match else name
