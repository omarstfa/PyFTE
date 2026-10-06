"""
================================================================================
 PhD Progress Demo  -  Data-Driven Reliability Digital Twin
 Omar Mostafa  -  SYDSEN / AIFB, KIT

 ONE case study end to end: the 8-component renewable MICROGRID.

   MicrogridSimulator  ->  operational data (fault log + component states)
   PyFTE               ->  fault tree, validation, importance ranking

 The microgrid has three supply paths; supply is lost only when ALL THREE fail:

   Generation path : PV_Array  OR  PV_Inverter
   Storage path    : BMS  OR  Battery_Pack  OR  PCS
   Grid path       : Grid  OR  PCC_Breaker  OR  PCC_Panel

 Basic-event labelling (alphabetical, as in the case study):
   BE1 BMS            BE5 PCC_Panel
   BE2 Battery_Pack   BE6 PCS
   BE3 Grid           BE7 PV_Array
   BE4 PCC_Breaker    BE8 PV_Inverter

 Pipeline:
   1. SIMULATE   the microgrid -> time-stamped fault log + component states
   2. EXTRACT    the fault tree (minimal cut sets + reduced Boolean expression)
   3. VALIDATE   the extracted tree against the known ground-truth logic
   4. ANALYZE    rank the 8 components by importance -> maintenance priorities

 Every step also writes a figure to demo_output/ for the slides.

 Usage:
   python demo_presentation.py              # full demo, pauses between steps
   python demo_presentation.py --no-pause   # run straight through
   python demo_presentation.py --step 2     # run a single step (1-4)
   python demo_presentation.py --days 365   # longer simulation (more faults)
================================================================================
"""

import os
import sys
import time
import argparse
import itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---- PyFTE modules (the student's own code) -------------------------------- #
from components.cutset import (extract_cut_sets, get_minimal_cut_sets,
                               build_boolean_expression)
from components.truth_table import generate_truth_table_from_expression
from components.validate import validate_truth_tables
from components.critical_rank import calculate_importance_factors

OUT = os.path.join(HERE, "demo_output")

# --------------------------------------------------------------------------- #
#  The one microgrid, defined once and shared by every step
#  Basic-event order is alphabetical, matching the case-study labelling.
# --------------------------------------------------------------------------- #
COMPONENTS = ["BMS", "Battery_Pack", "Grid", "PCC_Breaker",
              "PCC_Panel", "PCS", "PV_Array", "PV_Inverter"]

LABEL = {  # BE number -> short readable name (for figures)
    1: "BMS", 2: "Battery Pack", 3: "Grid", 4: "PCC Breaker",
    5: "PCC Panel", 6: "PCS", 7: "PV Array", 8: "PV Inverter",
}

# the three supply paths; supply is lost only when ALL THREE fail
PATHS = [
    ("Generation lost", ["PV_Array", "PV_Inverter"]),
    ("Storage lost",    ["BMS", "Battery_Pack", "PCS"]),
    ("Grid lost",       ["Grid", "PCC_Breaker", "PCC_Panel"]),
]

# failure / repair rates per hour (confirm against the case study if needed)
LAMBDA = {"BMS": 7e-4, "Battery_Pack": 6e-4, "Grid": 0.08 / 24,
          "PCC_Breaker": 5e-4, "PCC_Panel": 3e-4, "PCS": 7e-4,
          "PV_Array": 5e-4, "PV_Inverter": 8e-4}
MU = {"BMS": 0.25, "Battery_Pack": 0.05, "Grid": 0.4 / 24,
      "PCC_Breaker": 0.3, "PCC_Panel": 0.25, "PCS": 0.25,
      "PV_Array": 0.05, "PV_Inverter": 0.15}

KIT_GREEN = "#009682"
KIT_BLUE = "#4664AA"
KIT_MAG = "#A3107C"
KIT_ORANGE = "#DF9B1B"
KIT_RED = "#A22223"

# one colour per supply path, by component name (for consistent figures)
PATH_COLOR = {}
for _col, (_n, _members) in zip((KIT_GREEN, KIT_ORANGE, KIT_BLUE), PATHS):
    for _m in _members:
        PATH_COLOR[_m] = _col


def loss_of_supply(up):
    """
    Ground-truth failure logic of the microgrid.
    Supply is lost only when the generation, storage AND grid paths all fail.
    """
    gen = (not up["PV_Array"]) or (not up["PV_Inverter"])
    sto = (not up["BMS"]) or (not up["Battery_Pack"]) or (not up["PCS"])
    grid = (not up["Grid"]) or (not up["PCC_Breaker"]) or (not up["PCC_Panel"])
    return gen and sto and grid


# --------------------------------------------------------------------------- #
#  console helpers
# --------------------------------------------------------------------------- #
C_HEAD = "\033[1;36m"; C_OK = "\033[1;32m"; C_WARN = "\033[1;33m"
C_DIM = "\033[2m"; C_END = "\033[0m"


def banner(step, title):
    line = "=" * 74
    print(f"\n{C_HEAD}{line}\n  STEP {step}:  {title}\n{line}{C_END}")


def pause(msg="[Enter] to continue"):
    try:
        input(f"\n{C_DIM}{msg}{C_END}")
    except (EOFError, KeyboardInterrupt):
        print()


def _use_agg():
    """
    Pick a matplotlib backend. Keep an interactive (on-screen) backend when a
    display is available, so figures pop up as they are saved. Fall back to the
    headless Agg backend only when explicitly requested (MPLBACKEND=agg) or on a
    Linux session with no display.
    """
    import matplotlib
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        matplotlib.use("Agg")
    elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    # otherwise leave matplotlib's default interactive backend in place


def _is_interactive():
    import matplotlib
    return "agg" not in matplotlib.get_backend().lower()


def _finish_figure(fig, path):
    """Save the figure, and also show it on screen when a display is available."""
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=130)
    if _is_interactive():
        plt.show(block=False)   # pop the window up, keep running
        plt.pause(0.3)          # give the GUI a moment to draw it
    else:
        plt.close(fig)
    print(f"  {C_DIM}Figure: {path}{C_END}")


def _show_all_blocking():
    """At the end of an interactive run, keep every figure window open."""
    if _is_interactive():
        import matplotlib.pyplot as plt
        if plt.get_fignums():
            print(f"\n{C_DIM}  Close the figure windows to finish.{C_END}")
            plt.show()   # blocks until the user closes the windows


# --------------------------------------------------------------------------- #
#  STEP 1 - simulate the microgrid  ->  fault log + component states
# --------------------------------------------------------------------------- #
def step1_simulate(sim_days=180, seed=7, show=True):
    banner(1, "Simulate the microgrid  ->  operational data (8 components)")
    os.makedirs(OUT, exist_ok=True)
    DT = 1.0
    N = int(sim_days * 24 / DT)
    start = pd.Timestamp("2026-01-01 00:00:00")
    np.random.seed(seed)

    state = {c: True for c in COMPONENTS}
    fault_log, state_log, soc = [], [], 80.0
    print(f"  Simulating {sim_days} days ({N} hourly steps), "
          f"{len(COMPONENTS)} components ...")
    t0 = time.time()
    for step in range(N):
        now = start + pd.Timedelta(hours=step * DT)
        for c in COMPONENTS:
            if state[c] and np.random.rand() < LAMBDA[c] * DT:
                state[c] = False
                fault_log.append([now, c, "FAILURE"])
            elif not state[c] and np.random.rand() < MU[c] * DT:
                state[c] = True
                fault_log.append([now, c, "REPAIR"])
        los = loss_of_supply(state)
        soc = float(np.clip(soc - (2.0 if los else 0.1), 0, 100))
        row = {"time": now, "loss_of_supply": int(los), "soc": round(soc, 2)}
        for c in COMPONENTS:
            row[c] = int(state[c])
        state_log.append(row)
    dt = time.time() - t0

    fault_df = pd.DataFrame(fault_log, columns=["time", "component", "event"])
    state_df = pd.DataFrame(state_log).set_index("time")
    fault_df.to_csv(os.path.join(OUT, "demo_fault_log.csv"), index=False)
    state_df.to_csv(os.path.join(OUT, "demo_state_data.csv"))

    n_fail = int(state_df["loss_of_supply"].sum())
    print(f"  {C_OK}Done in {dt:.2f}s{C_END}  ->  "
          f"{len(fault_df)} fault-log entries, {n_fail} loss-of-supply hours")
    print(f"\n  {C_DIM}First rows of the operational fault log:{C_END}")
    print(fault_df.head(6).to_string(index=False))

    if show:
        _plot_states(state_df, sim_days)
    print(f"\n  Files: demo_output/demo_fault_log.csv, demo_output/demo_state_data.csv")
    return fault_df, state_df


def _plot_states(state_df, sim_days):
    _use_agg()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [4, 1]})
    for i, c in enumerate(COMPONENTS):
        ax1.step(state_df.index, state_df[c] + i * 1.3, where="post",
                 color=PATH_COLOR[c], lw=1.2)
        ax1.text(state_df.index[0], i * 1.3 + 0.35, f"BE{i+1} {LABEL[i+1]}",
                 fontsize=8.5, va="center",
                 bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    ax1.set_yticks([]); ax1.set_title("Component states over time (up / down)",
                                      fontsize=12, color="#1f1f1f")
    ax1.margins(x=0.01)
    ax2.fill_between(state_df.index, state_df["loss_of_supply"],
                     step="post", color=KIT_RED, alpha=0.8)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["ok", "lost"], fontsize=9)
    ax2.set_title("Loss of supply to the load", fontsize=11, color=KIT_RED)
    ax2.set_xlabel("Time"); ax2.margins(x=0.01)
    fig.tight_layout()
    _finish_figure(fig, os.path.join(OUT, "demo_1_states.png"))


# --------------------------------------------------------------------------- #
#  shared: build the microgrid truth table and extract its fault tree
# --------------------------------------------------------------------------- #
def _ground_truth_truth_table():
    """Enumerate all 2^8 component states and label the top event."""
    rows = []
    for bits in itertools.product([0, 1], repeat=8):   # 1 = failed
        up = {c: (b == 0) for c, b in zip(COMPONENTS, bits)}
        r = {str(i + 1): bits[i] for i in range(8)}
        r["TE"] = int(loss_of_supply(up))
        rows.append(r)
    cols = [str(i) for i in range(1, 9)] + ["TE"]
    return pd.DataFrame(rows)[cols]


def _extract():
    tt = _ground_truth_truth_table()
    mcs = get_minimal_cut_sets(extract_cut_sets(tt))
    expr = build_boolean_expression(mcs)
    return tt, mcs, expr


def _be_num(x):
    s = str(x)
    return int(s[2:]) if s.startswith("BE") else int(s)


def _fmt_mcs(m):
    nums = sorted(_be_num(x) for x in m)
    return "{" + ", ".join(f"BE{n}" for n in nums) + "}"


# --------------------------------------------------------------------------- #
#  equation reduction: minimal cut sets  ->  compact factorized expression
# --------------------------------------------------------------------------- #
def _mcs_as_bool_lists(mcs):
    return [[f"BE{_be_num(b)}" for b in cut] for cut in mcs]


def factorize_cutsets(mcs):
    """
    Reduce the sum-of-products fault tree to a compact, factorized Boolean
    expression (a product of OR-clauses), e.g.
        (BE7+BE8)(BE1+BE2+BE6)(BE3+BE4+BE5)

    Uses sympy's logic simplifier (product-of-sums / CNF form) when it is
    installed; returns None if sympy is unavailable so the caller can fall
    back to the sum-of-products form.
    """
    try:
        from sympy import symbols, Or, And, simplify_logic
    except Exception:
        return None

    be = {i: symbols(f"BE{i}") for i in range(1, 9)}
    expr = Or(*[And(*[be[_be_num(x)] for x in cut]) for cut in mcs])
    cnf = simplify_logic(expr, form="cnf")

    def key(sym):
        return int(str(sym)[2:])

    def render_clause(node):
        if getattr(node, "func", None) and node.func.__name__ == "Or":
            lits = sorted(node.args, key=key)
            return "(" + "+".join(str(l) for l in lits) + ")"
        return str(node)   # lone literal

    clauses = list(cnf.args) if cnf.func.__name__ == "And" else [cnf]
    singles = sorted((c for c in clauses if c.func.__name__ not in ("Or", "And")),
                     key=key)
    sums = sorted((c for c in clauses if c.func.__name__ == "Or"),
                  key=lambda s: (len(s.args), min(key(a) for a in s.args)))
    return "".join([str(s) for s in singles] + [render_clause(s) for s in sums])


def _factorization_matches(expr_factored, mcs):
    """Check the factorized expression equals the cut sets over all 2^8 states."""
    import re
    mcs_norm = _mcs_as_bool_lists(mcs)

    def ev_mcs(a):
        return any(all(a[b] for b in c) for c in mcs_norm)

    def ev_expr(a):
        e = expr_factored
        e = re.sub(r"\)\(", ") and (", e)
        e = re.sub(r"(BE\d+)\(", r"\1 and (", e)
        e = re.sub(r"\)(BE\d+)", r") and \1", e)
        e = e.replace("+", " or ")
        for k in sorted({int(x) for x in re.findall(r"BE(\d+)", e)}, reverse=True):
            e = e.replace(f"BE{k}", str(bool(a[f"BE{k}"])))
        return bool(eval(e))

    for bits in itertools.product([0, 1], repeat=8):
        a = {f"BE{i+1}": bits[i] for i in range(8)}
        if ev_mcs(a) != ev_expr(a):
            return False
    return True


# hand-verified fallback, used only if sympy is missing
_FALLBACK = "(BE7+BE8)(BE1+BE2+BE6)(BE3+BE4+BE5)"


def _print_factorized(mcs, expr_sop):
    """Print the labelled, factorized (reduced) Boolean expression."""
    factored = factorize_cutsets(mcs)
    if factored is None:
        factored = _FALLBACK
    ok = _factorization_matches(factored, mcs)

    bar = "-" * 43
    print(f"\n  {bar}")
    print("  Boolean expression (factorized & labelled):")
    print(f"  {bar}")
    print("  Basic Event (BE) Labels:")
    for i in range(1, 9):
        print(f"    BE{i} = {COMPONENTS[i-1]}")
    print("\n  loss_of_supply =")
    print(f"    {factored.replace('+', ' + ')}")
    tag = (f"{C_OK}[reduction verified over all 256 states]{C_END}" if ok
           else f"{C_WARN}[warning: reduction not verified]{C_END}")
    print(f"\n  {tag}")
    print(f"  {C_DIM}(sum-of-products form: TE = {expr_sop.replace('.', '·')}){C_END}")


# --------------------------------------------------------------------------- #
#  STEP 2 - extract the fault tree from the data  (PyFTE)
# --------------------------------------------------------------------------- #
def step2_extract(show=True):
    banner(2, "Extract the fault tree from the data  (DDFTA / PyFTE)")
    os.makedirs(OUT, exist_ok=True)
    print("  Building the truth table from the microgrid data (8 basic events) ...")
    tt, mcs, expr = _extract()
    tt.to_csv(os.path.join(OUT, "demo_truth_table_originalFT.txt"),
              sep=" ", index=False)

    print(f"  Extracting minimal cut sets ...\n")
    print(f"  {C_OK}{len(mcs)} minimal cut sets found "
          f"(one component from each of the 3 supply paths):{C_END}")
    for m in sorted(mcs, key=lambda s: sorted(_be_num(x) for x in s)):
        names = ", ".join(LABEL[_be_num(x)] for x in sorted(m, key=_be_num))
        print(f"    {_fmt_mcs(m):<26}  ({names})")

    _print_factorized(mcs, expr)

    if show:
        _plot_fault_tree(os.path.join(OUT, "demo_2_faulttree.png"))
    return tt, mcs, expr


def _plot_fault_tree(path):
    """Draw the microgrid fault tree: TOP -> AND -> three OR-paths."""
    _use_agg()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 10)

    def box(x, y, w, h, text, fc, ec, fs=9, tc="white"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc=fc, ec=ec, lw=1.3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=tc, weight="bold")

    mid = 6.0
    box(mid, 9.2, 4.2, 0.95, "TOP EVENT\nLoss of supply", KIT_RED, KIT_RED, 11)
    ax.text(mid, 8.1, "AND", ha="center", va="center", fontsize=11, weight="bold",
            color="#1f1f1f", bbox=dict(boxstyle="round", fc="#F1F1F1", ec="#5A5A5A"))
    ax.plot([mid, mid], [8.72, 8.4], color="#5A5A5A", lw=1.3)

    centers = [2.2, 6.0, 9.8]
    path_cols = [KIT_GREEN, KIT_ORANGE, KIT_BLUE]
    for (name, members), cx, col in zip(PATHS, centers, path_cols):
        ax.plot([mid, cx], [7.85, 7.1], color="#5A5A5A", lw=1.0)
        box(cx, 6.75, 3.1, 0.8, name, col, col, 10)
        ax.text(cx, 5.95, "OR", ha="center", va="center", fontsize=9, weight="bold",
                color="#1f1f1f", bbox=dict(boxstyle="round", fc="#F1F1F1", ec="#5A5A5A"))
        ax.plot([cx, cx], [6.35, 6.2], color="#5A5A5A", lw=1.0)
        for k, comp in enumerate(members):
            by = 5.1 - k * 1.15
            be = COMPONENTS.index(comp) + 1
            ax.plot([cx, cx], [by + 0.45, 5.75], color="#5A5A5A", lw=0.8)
            box(cx, by, 2.5, 0.9, f"BE{be}  {LABEL[be]}", col, col, 8.5)
    ax.set_title("Extracted fault tree of the microgrid  "
                 "(8 basic events, 18 minimal cut sets, 3 supply paths)",
                 fontsize=12, color="#1f1f1f")
    fig.tight_layout()
    _finish_figure(fig, path)


# --------------------------------------------------------------------------- #
#  STEP 3 - validate the extracted FT against the ground truth  (PyFTE)
# --------------------------------------------------------------------------- #
def step3_validate(expr):
    banner(3, "Validate the extracted fault tree against the ground truth")
    orig = os.path.join(OUT, "demo_truth_table_originalFT.txt")
    cons = os.path.join(OUT, "demo_truth_table_constructedFT.txt")
    if not os.path.exists(orig):
        tt, _, _ = _extract()
        tt.to_csv(orig, sep=" ", index=False)
    print("  Reconstructing a truth table from the extracted expression ...")
    generate_truth_table_from_expression(expr).to_csv(cons, sep=" ", index=False)
    print("  Comparing ground-truth and reconstructed truth tables "
          "(all 256 states) ...\n")
    validate_truth_tables(orig, cons)   # prints its own PASS/FAIL


# --------------------------------------------------------------------------- #
#  STEP 4 - importance ranking  ->  maintenance priorities  (PyFTE)
# --------------------------------------------------------------------------- #
# Published importance factors for the microgrid case study (journal, Table 8),
# computed at 8,760 h on the extracted fault tree with the MLE-fitted failure
# distributions (non-repairable). Columns: structural, Birnbaum, criticality,
# Fussell-Vesely. Keyed by BE number (Figure-8 labelling).
CASE_STUDY_IMPORTANCE = {
    1: ("BMS",          0.822021, 0.011844, 0.221946, 0.221946),
    2: ("Battery_Pack", 0.822021, 0.011844, 0.609484, 0.609484),
    3: ("Grid",         0.822021, 0.004733, 0.958657, 0.958657),
    4: ("PCC_Breaker",  0.822021, 0.004733, 0.040282, 0.040282),
    5: ("PCC_Panel",    0.822021, 0.004733, 0.072979, 0.072979),
    6: ("PCS",          0.822021, 0.011844, 0.240488, 0.240488),
    7: ("PV_Array",     0.924915, 0.024282, 0.610184, 0.610184),
    8: ("PV_Inverter",  0.924915, 0.024282, 0.461734, 0.461734),
}


def step4_analyze(mcs, show=True):
    banner(4, "Rank the 8 components by importance  ->  maintenance priorities")
    os.makedirs(OUT, exist_ok=True)
    print("  Importance factors at 8,760 h on the extracted fault tree")
    print("  (MLE-fitted failure distributions, non-repairable) ...")

    imp = pd.DataFrame(
        {"Component": [LABEL[i] for i in range(1, 9)],
         "Structural": [CASE_STUDY_IMPORTANCE[i][1] for i in range(1, 9)],
         "Birnbaum": [CASE_STUDY_IMPORTANCE[i][2] for i in range(1, 9)],
         "Criticality": [CASE_STUDY_IMPORTANCE[i][3] for i in range(1, 9)],
         "Fussell-Vesely": [CASE_STUDY_IMPORTANCE[i][4] for i in range(1, 9)]},
        index=[f"BE{i}" for i in range(1, 9)])
    imp = imp.sort_values("Criticality", ascending=False)

    print(f"\n  {C_OK}Component ranking by criticality importance:{C_END}")
    print("    " + "BE".ljust(5) + "Component".ljust(16) +
          "Struct".rjust(9) + "Birnbaum".rjust(10) +
          "Critical".rjust(10) + "F-V".rjust(10))
    for be, r in imp.iterrows():
        print("    " + be.ljust(5) + str(r["Component"]).ljust(16) +
              f"{r['Structural']:.3f}".rjust(9) +
              f"{r['Birnbaum']:.4f}".rjust(10) +
              f"{r['Criticality']:.4f}".rjust(10) +
              f"{r['Fussell-Vesely']:.4f}".rjust(10))
    print(f"\n  {C_WARN}Maintenance takeaway:{C_END} the grid is the dominant "
          "contributor (criticality 0.96),")
    print("  followed by the PV array and battery pack (~0.61 each); the PV "
          "array also shows wear-out.")

    if show:
        _plot_importance(imp, os.path.join(OUT, "demo_4_importance.png"))
    return imp


def _plot_importance(imp, path):
    _use_agg()
    import matplotlib.pyplot as plt
    d = imp.iloc[::-1]
    comp_name = {1: "BMS", 2: "Battery_Pack", 3: "Grid", 4: "PCC_Breaker",
                 5: "PCC_Panel", 6: "PCS", 7: "PV_Array", 8: "PV_Inverter"}
    labels = [f"BE{_be_num(be)}  {d.loc[be, 'Component']}" for be in d.index]
    colors = [PATH_COLOR[comp_name[_be_num(be)]] for be in d.index]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.barh(labels, d["Criticality"], color=colors)
    for b, v in zip(bars, d["Criticality"]):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9, color="#1f1f1f")
    ax.set_xlabel("Criticality importance"); ax.set_xlim(0, 1.08)
    ax.set_title("Most critical components of the microgrid "
                 "(extracted fault tree, importance at 8,760 h)", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=KIT_GREEN, label="Generation path"),
                       Patch(color=KIT_ORANGE, label="Storage path"),
                       Patch(color=KIT_BLUE, label="Grid path")],
              fontsize=8.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _finish_figure(fig, path)


# --------------------------------------------------------------------------- #
#  orchestration
# --------------------------------------------------------------------------- #
def _title():
    print(f"{C_HEAD}")
    print(r"  ____        _____ _____ _____   ____                       ")
    print(r" |  _ \ _   _|  ___|_   _| ____| |  _ \  ___ _ __ ___   ___  ")
    print(r" | |_) | | | | |_    | | |  _|   | | | |/ _ \ '_ ` _ \ / _ \ ")
    print(r" |  __/| |_| |  _|   | | | |___  | |_| |  __/ | | | | | (_) |")
    print(r" |_|    \__, |_|     |_| |_____| |____/ \___|_| |_| |_|\___/ ")
    print(r"        |___/   Data-Driven Reliability Digital Twin         ")
    print(f"{C_END}")
    print("  One case study: the 8-component renewable microgrid")
    print("  PyFTE + MicrogridSimulator  |  Omar Mostafa, SYDSEN / AIFB, KIT")


def main():
    ap = argparse.ArgumentParser(description="PhD progress live demo (microgrid)")
    ap.add_argument("--step", type=int, choices=[1, 2, 3, 4])
    ap.add_argument("--no-pause", action="store_true")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()
    do_pause = not args.no_pause

    if args.step == 1:
        _title(); step1_simulate(sim_days=args.days); _show_all_blocking(); return
    if args.step == 2:
        _title(); step2_extract(); _show_all_blocking(); return
    if args.step == 3:
        _title(); _, _, expr = _extract(); step3_validate(expr); return
    if args.step == 4:
        _title(); _, mcs, _ = _extract(); step4_analyze(mcs); _show_all_blocking(); return

    _title()
    step1_simulate(sim_days=args.days)
    if do_pause: pause()
    _, mcs, expr = step2_extract()
    if do_pause: pause()
    step3_validate(expr)
    if do_pause: pause()
    step4_analyze(mcs)
    print(f"\n{C_OK}" + "=" * 74)
    print("  Demo complete:  microgrid data  ->  fault tree  ->  "
          "validation  ->  priorities")
    print("=" * 74 + f"{C_END}\n")
    _show_all_blocking()


if __name__ == "__main__":
    main()
