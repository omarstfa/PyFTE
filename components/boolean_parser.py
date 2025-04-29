class Node:
    def __init__(self, name, node_type="Basic Events", children=None):
        self.name = name
        self.node_type = node_type
        self.children = children if children else []

def parse_expression(expr: str) -> Node:
    terms = [term.strip() for term in expr.split('+')]
    children = []
    for term in terms:
        factors = term.split('.')
        if len(factors) == 1:
            children.append(Node(factors[0], node_type="Basic Events"))
        else:
            and_children = [Node(f.strip(), node_type="Basic Events") for f in factors]
            and_node = Node("AND", node_type="AND", children=and_children)
            children.append(and_node)
    return Node("Top Event", node_type="OR", children=children)

def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node.name} ({node.node_type})")
    for child in node.children:
        print_tree(child, level + 1)
