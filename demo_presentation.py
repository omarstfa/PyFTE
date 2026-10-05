"""
================================================================================
 PhD Progress Demo  -  Data-Driven Reliability Digital Twin
 Omar Mostafa  -  SYDSEN / AIFB, KIT

 ONE case study end to end: the 8-component renewable MICROGRID.

   MicrogridSimulator  ->  operational data (fault log + component states)
   PyFTE               ->  fault tree, validation, importance ranking

 The same 8 basic events are used from the first step to the last:

   BE1 Grid                 BE5 PV Inverter
   BE2 PCC Breaker          BE6 Battery Pack
   BE3 Islanding Controller BE7 BMS
   BE4 PV Array             BE8 PCS

 Pipeline:
   1. SIMULATE   the microgrid -> time-stamped fault log + component states
   2. EXTRACT    the fault tree (minimal cut sets + Boolean expression) from data
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
# --------------------------------------------------------------------------- #
COMPONENTS = ["Grid", "PCC_Breaker", "Islanding_Controller", "PV_Array",
              "PV_Inverter", "Battery_Pack", "BMS", "PCS"]

LABEL = {  # BE number -> short readable name
    1: "Grid", 2: "PCC Breaker", 3: "Islanding Ctrl", 4: "PV Array",
    5: "PV Inverter", 6: "Battery Pack", 7: "BMS", 8: "PCS",
}

# failure / repair rates per hour, exactly as in microgird_sim.py
LAMBDA = {"Grid": 0.08 / 24, "PCC_Breaker": 5e-4, "Islanding_Controller": 7e-4,
          "PV_Array": 5e-4, "PV_Inverter": 8e-4, "Battery_Pack": 6e-4,
          "BMS": 7e-4, "PCS": 7e-4}
MU = {"Grid": 0.4 / 24, "PCC_Breaker": 0.3, "Islanding_Controller": 0.25,
      "PV_Array": 0.05, "PV_Inverter": 0.15, "Battery_Pack": 0.05,
      "BMS": 0.25, "PCS": 0.25}

KIT_GREEN = "#009682"
KIT_BLUE = "#4664AA"
KIT_MAG = "#A3107C"
KIT_ORANGE = "#DF9B1B"
KIT_RED = "#A22223"


def loss_of_supply(up):
    """Ground-truth failure logic of the microgrid (from microgird_sim.py)."""
    BE1 = not up["Grid"]
    BE2 = not up["PCC_Breaker"]
    BE3 = not up["Islanding_Controller"]
    BE4 = not up["PV_Array"]
    BE5 = not up["PV_Inverter"]
    BE6 = not up["Battery_Pack"]
    BE7 = not up["BMS"]
    BE8 = not up["PCS"]
    immediate_failure = BE1 and (BE2 or BE3)
    islanded_failure = BE1 and ((BE4 or BE5) and (BE6 or BE7 or BE8))
    return immediate_failure or islanded_failure


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
    import matplotlib
    if os.environ.get("MPLBACKEND", "").lower() == "agg" or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")


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
    palette = [KIT_GREEN, KIT_BLUE, KIT_MAG, KIT_ORANGE,
               "#5A9BD4", "#70AD47", "#7B1FA2", KIT_RED]
    for i, c in enumerate(COMPONENTS):
        ax1.step(state_df.index, state_df[c] + i * 1.3, where="post",
                 color=palette[i], lw=1.2)
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
    p = os.path.join(OUT, "demo_1_states.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  {C_DIM}Figure: {p}{C_END}")


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


def _fmt_mcs(m):
    nums = sorted(int(x) for x in m)
    return "{" + ", ".join(f"BE{n}" for n in nums) + "}"


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
          f"(every one contains BE1 Grid):{C_END}")
    for m in sorted(mcs, key=lambda s: (len(s), sorted(int(x) for x in s))):
        names = ", ".join(LABEL[int(x)] for x in sorted(m, key=int))
        print(f"    {_fmt_mcs(m):<22}  ({names})")
    print(f"\n  {C_DIM}Extracted Boolean expression:{C_END}")
    print("    TE = " + expr.replace(".", "·"))

    if show:
        _plot_fault_tree(mcs, os.path.join(OUT, "demo_2_faulttree.png"))
    return tt, mcs, expr


def _plot_fault_tree(mcs, path):
    """Draw the microgrid fault tree: TE -> OR -> cut sets."""
    _use_agg()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    mcs_sorted = sorted(mcs, key=lambda s: (len(s), sorted(int(x) for x in s)))
    n = len(mcs_sorted)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off"); ax.set_xlim(0, n); ax.set_ylim(0, 10)

    def box(x, y, w, h, text, fc, ec, fs=9, tc="white"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=fc, ec=ec, lw=1.3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=tc, weight="bold")

    midx = n / 2.0
    box(midx, 9.1, 3.2, 1.0, "TOP EVENT\nLoss of supply", KIT_RED, KIT_RED, 10)
    ax.text(midx, 7.9, "OR", ha="center", va="center", fontsize=11,
            weight="bold", color="#1f1f1f",
            bbox=dict(boxstyle="round", fc="#F1F1F1", ec="#5A5A5A"))
    ax.plot([midx, midx], [8.6, 8.2], color="#5A5A5A", lw=1.2)
    for j, m in enumerate(mcs_sorted):
        cx = j + 0.5
        ax.plot([midx, cx], [7.6, 6.7], color="#5A5A5A", lw=1.0)
        nums = sorted(int(x) for x in m)
        if len(nums) > 1:
            ax.text(cx, 6.5, "AND", ha="center", va="center", fontsize=8,
                    weight="bold", bbox=dict(boxstyle="round", fc="#E8ECF6",
                                             ec=KIT_BLUE))
        for k, be in enumerate(nums):
            by = 5.6 - k * 1.25
            col = KIT_RED if be == 1 else (KIT_BLUE if be in (2, 3) else KIT_GREEN)
            box(cx, by, 0.92, 1.0, f"BE{be}\n{LABEL[be]}", col, col, 7.0)
            top_y = 6.2 if len(nums) > 1 else 6.7
            ax.plot([cx, cx], [by + 0.5, top_y], color="#5A5A5A", lw=0.8)
    ax.set_title("Extracted fault tree of the microgrid  "
                 "(8 basic events, 8 minimal cut sets)",
                 fontsize=12, color="#1f1f1f")
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  {C_DIM}Figure: {path}{C_END}")


# --------------------------------------------------------------------------- #
#  STEP 3 - validate the extracted FT against the ground truth  (PyFTE)
# --------------------------------------------------------------------------- #
def step3_validate(expr):
    banner(3, "Validate the extracted fault tree against the ground truth")
    orig = os.path.join(OUT, "demo_truth_table_originalFT.txt")
    cons = os.path.join(OUT, "demo_truth_table_constructedFT.txt")
    if not os.path.exists(orig):
        _extract_and_save_tt(orig)
    print("  Reconstructing a truth table from the extracted expression ...")
    generate_truth_table_from_expression(expr).to_csv(cons, sep=" ", index=False)
    print("  Comparing ground-truth and reconstructed truth tables "
          "(all 256 states) ...\n")
    validate_truth_tables(orig, cons)   # prints its own PASS/FAIL


def _extract_and_save_tt(path):
    tt, _, _ = _extract()
    tt.to_csv(path, sep=" ", index=False)


# --------------------------------------------------------------------------- #
#  STEP 4 - importance ranking  ->  maintenance priorities  (PyFTE)
# --------------------------------------------------------------------------- #
def step4_analyze(mcs, show=True):
    banner(4, "Rank the 8 components by importance  ->  maintenance priorities")
    T = 8760.0   # 1-year mission
    mcs_BE = [[f"BE{int(x)}" for x in cut] for cut in mcs]
    label_map = {f"BE{i}": i - 1 for i in range(1, 9)}
    failure_probs = {f"BE{i}": 1 - np.exp(-LAMBDA[COMPONENTS[i - 1]] * T)
                     for i in range(1, 9)}

    print("  Computing Birnbaum, criticality and Fussell-Vesely importance "
          "(1-year mission) ...")
    imp = calculate_importance_factors(mcs_BE, label_map, failure_probs)
    imp["Component"] = [LABEL[int(be[2:])] for be in imp.index]
    imp = imp.sort_values("Criticality", ascending=False)

    print(f"\n  {C_OK}Component ranking by criticality importance:{C_END}")
    print("    " + "BE".ljust(5) + "Component".ljust(17) +
          "Criticality".rjust(12) + "Birnbaum".rjust(11))
    for be, r in imp.iterrows():
        print("    " + be.ljust(5) + str(r["Component"]).ljust(17) +
              f"{r['Criticality']:.3f}".rjust(12) +
              f"{r['Birnbaum']:.2f}".rjust(11))
    print(f"\n  {C_WARN}Maintenance takeaway:{C_END} the grid (BE1) is in every "
          "cut set, so grid resilience comes first;")
    print("  then the PV path (inverter, array), then the storage chain "
          "(battery, BMS, PCS).")

    if show:
        _plot_importance(imp, os.path.join(OUT, "demo_4_importance.png"))
    return imp


def _plot_importance(imp, path):
    _use_agg()
    import matplotlib.pyplot as plt
    d = imp.iloc[::-1]
    labels = [f"BE{be[2:]}  {d.loc[be, 'Component']}" for be in d.index]
    colors = [KIT_RED if be == "BE1" else
              (KIT_BLUE if be in ("BE2", "BE3") else KIT_GREEN) for be in d.index]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.barh(labels, d["Criticality"], color=colors)
    for b, v in zip(bars, d["Criticality"]):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9, color="#1f1f1f")
    ax.set_xlabel("Criticality importance"); ax.set_xlim(0, 1.12)
    ax.set_title("Most critical components of the microgrid "
                 "(extracted fault tree, 1-year mission)", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=KIT_RED, label="Grid (in every cut set)"),
                       Patch(color=KIT_BLUE, label="Immediate-failure path"),
                       Patch(color=KIT_GREEN, label="Islanded-failure path")],
              fontsize=8.5, loc="lower right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  {C_DIM}Figure: {path}{C_END}")


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
        _title(); step1_simulate(sim_days=args.days); return
    if args.step == 2:
        _title(); step2_extract(); return
    if args.step == 3:
        _title(); _, _, expr = _extract(); step3_validate(expr); return
    if args.step == 4:
        _title(); _, mcs, _ = _extract(); step4_analyze(mcs); return

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


if __name__ == "__main__":
    main()
