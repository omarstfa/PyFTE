from anytree.exporter import DotExporter

GATES = ['AND', 'OR', 'VOTING']


def node_name(node):
    if node.name in GATES:
        return '%s' % node.id
    else:
        return '%s' % node.name


def node_attr(node):
    if node.name in GATES:
        if node.name == 'VOTING':
            return 'color="red" shape=house label="%s/%s %s"' % (node.k, len(node.children), node.name)
        if node.name == 'AND':
            return 'color="red" shape=house label="%s"' % node.name
        if node.name == 'OR':
            return 'color="red" shape=house label="%s"' % node.name
    else:
        return 'color="blue" shape=box'


def edge_type(node, child):
    return '--'


def export_to_dot(fault_tree, file_name):
    DotExporter(fault_tree.top_event, graph="graph", nodenamefunc=node_name, nodeattrfunc=node_attr,
                edgetypefunc=edge_type).to_dotfile(file_name)


def export_to_png(fault_tree, file_name):
    DotExporter(fault_tree.top_event, graph="graph", nodenamefunc=node_name, nodeattrfunc=node_attr,
                edgetypefunc=edge_type).to_picture(file_name)
