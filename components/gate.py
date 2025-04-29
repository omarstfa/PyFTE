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
