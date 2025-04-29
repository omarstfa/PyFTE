import pandas as pd

def extract_cut_sets(df):
    basic_events = df.columns[:-1]
    TE = df.columns[-1]
    cut_sets = []

    for _, row in df.iterrows():
        if row[TE] == 1:
            cut_sets.append(set(basic_events[row[basic_events] == 1]))

    return cut_sets

def get_minimal_cut_sets(cut_sets):
    cut_sets = sorted(cut_sets, key=lambda x: len(x))
    minimal_cut_sets = []
    for cs in cut_sets:
        if not any(cs.issuperset(mcs) for mcs in minimal_cut_sets):
            minimal_cut_sets.append(cs)
    return minimal_cut_sets

def build_boolean_expression(minimal_cut_sets):
    return " + ".join(["·".join(sorted(mcs)) for mcs in minimal_cut_sets])
