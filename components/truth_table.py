import itertools
import pandas as pd
import re

def extract_variables(expr):
    return sorted(set(re.findall(r'\b\d+\b', expr)), key=lambda x: int(x))

def generate_truth_table_from_expression(expr: str):
    variables = extract_variables(expr)
    table = []

    for combo in itertools.product([0, 1], repeat=len(variables)):
        local_vars = {var: str(val) for var, val in zip(variables, combo)}
        expr_eval = expr
        for var, val in local_vars.items():
            expr_eval = re.sub(rf'\b{re.escape(var)}\b', val, expr_eval)
        expr_eval = expr_eval.replace('+', ' or ').replace('·', ' and ')
        result = eval(expr_eval)
        table.append([int(local_vars[var]) for var in variables] + [int(result)])

    return pd.DataFrame(table, columns=variables + ['TE'])
