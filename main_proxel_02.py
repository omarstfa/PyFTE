import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Components ---
from components.event import Event, extract_event_number
from components.gate import Gate
from components.fault_tree import FaultTree, print_fault_tree
from components.cutset import extract_cut_sets, get_minimal_cut_sets, build_boolean_expression
from components.truth_table import generate_truth_table_from_expression
from components.fault_log_generator import generate_synthetic_fault_logs_accelerated, fault_logs_to_truth_table
from components.validate import validate_truth_tables
from components.critical_rank import calculate_importance_factors, simulate_reliability, make_reliability_function, fit_exponential_reliability
from components.proxel_sim import BasicEvent, proxel_system   # new module

#%% ============================
# 1) Build A Fault Tree (FT)
# ============================
topEvent = Event('Top Event')
or0 = Gate('OR', parent=topEvent)

intermediateEvent1 = Event('Intermediate Event 1', parent=or0)
intermediateEvent2 = Event('Intermediate Event 2', parent=or0)

or1 = Gate('OR', parent=intermediateEvent1)
Event('Basic Event 1', parent=or1)
Event('Basic Event 2', parent=or1)

intermediateEvent3 = Event('Intermediate Event 3', parent=or1)
or2 = Gate('OR', parent=intermediateEvent3)
intermediateEvent4 = Event('Intermediate Event 4', parent=or2)
intermediateEvent5 = Event('Intermediate Event 5', parent=or2)
or3 = Gate('OR', parent=intermediateEvent4)

for i in range(5, 17):
    Event(f'Basic Event {i}', parent=or3)

and1 = Gate('AND', parent=intermediateEvent5)
Event('Basic Event 3', parent=and1)
Event('Basic Event 4', parent=and1)

or4 = Gate('OR', parent=intermediateEvent2)
Event('Basic Event 17', parent=or4)
Event('Basic Event 18', parent=or4)

print("\nOriginal Fault Tree Structure:")
print_fault_tree(topEvent)

# Truth table for the original FT
tree_original = FaultTree(topEvent)
truth_table_original = tree_original.generate_truth_table()
tt_simple = truth_table_original.copy()
column_mapping = {name: extract_event_number(name) for name in tt_simple.columns}
column_mapping["Top Event"] = "TE"
tt_simple.rename(columns=column_mapping, inplace=True)
sorted_cols = sorted([c for c in tt_simple.columns if c != "TE"], key=lambda x: int(x)) + ["TE"]
tt_simple = tt_simple[sorted_cols]
tt_simple.to_csv("output/truth_table_originalFT.txt", sep=" ", index=False)

# --- Minimal cut sets & Boolean expression ---
cut_sets = extract_cut_sets(tt_simple)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)
boolean_expression = build_boolean_expression(minimal_cut_sets)

print("\nMinimal Cut Sets:")
for mcs in minimal_cut_sets:
    print(sorted(mcs))

print("\nGround-Truth Fault Tree Boolean Expression: TE =", boolean_expression.replace('.', '·'))

#%% Fault Log Generation
# ============================
# 2) Inputs (time, # of replications, failure rates, repair times)
# ============================

T_mission = 10000  # hours
N_SIM = 3000       # replications

failure_rates = {
    'BE1':1e-7, 'BE2':2e-7,
    'BE3':8e-7, 'BE4':8e-7,
    'BE5':8.46e-6, 'BE6':4.9e-6,
    'BE7':3e-7, 'BE8':1.3e-6,
    'BE9':8.8e-6, 'BE10':1.115e-5,
    'BE11':1.487e-5,'BE12':1.01e-6,
    'BE13':2.1e-7,'BE14':5.2e-7,
    'BE15':7.29e-6,'BE16':5.7e-6,
    'BE17':1e-7,'BE18':2e-7
}

# Multiplication factor of failure rates for faster results
x = 100
for k in failure_rates:
    failure_rates[k] *= x

repair_times = {
    'BE1': 4,   'BE2': 4,
    'BE3': 10,  'BE4': 10,
    'BE5': 24,  'BE6': 24,
    'BE7': 24,  'BE8': 8,
    'BE9': 8,   'BE10': 48,
    'BE11': 48, 'BE12': 24,
    'BE13': 24, 'BE14': 12,
    'BE15': 48, 'BE16': 48,
    'BE17': 4,  'BE18': 4,
}

# ============================
# Synthetic Fault Log Generation with Acceleration
# ============================

def add_fault_log_generation_to_main_accelerated(acceleration_factor=1000):
    # Generate synthetic fault logs with acceleration
    print("Generating accelerated synthetic fault logs...")
    fault_logs, be_descriptions, accelerated_rates = generate_synthetic_fault_logs_accelerated(
        failure_rates, repair_times, minimal_cut_sets,  # Pass original format
        mission_time_hours=T_mission,
        acceleration_factor=acceleration_factor,
        seed=42
    )
    # Convert to truth table
    truth_table = fault_logs_to_truth_table(fault_logs, be_descriptions, minimal_cut_sets)
    # Save results
    fault_logs.to_csv("output/synthetic_fault_logs_accelerated.csv", index=False)
    truth_table.to_csv("output/truth_table_from_accelerated_logs.csv", index=False)
    print(f"Generated {len(fault_logs)} fault log entries")
    print(f"Created truth table with {len(truth_table)} unique state combinations")
    print(f"System failure occurred in {fault_logs['System_Failure'].sum()} events")
    # Display statistics
    print("\nFault Statistics:")
    for be in sorted(failure_rates.keys()):
        be_faults = len(fault_logs[(fault_logs['Basic_Event'] == be) & (fault_logs['Status'] == 'Active')])
        print(f"  {be}: {be_faults} faults")
    # Display sample of fault logs
    print("\nSample of synthetic fault logs:")
    sample_logs = fault_logs.head(10)[['Timestamp', 'Description', 'Status', 'System_Failure']]
    print(sample_logs.to_string(index=False))
    return fault_logs, truth_table, accelerated_rates

def validate_and_compare_fault_trees(original_minimal_cut_sets, truth_table_from_logs):
    """
    Validate that the generated truth table produces the same minimal cut sets
    """
    print("\n" + "="*60)
    print("FAULT TREE VALIDATION")
    print("="*60)

    # Extract cut sets from generated truth table
    cut_sets_from_logs = extract_cut_sets(truth_table_from_logs)
    minimal_cut_sets_from_logs = get_minimal_cut_sets(cut_sets_from_logs)
    boolean_expression_from_logs = build_boolean_expression(minimal_cut_sets_from_logs)

    # Convert both to standardized format for comparison
    def standardize_mcs(mcs_list):
        """Convert all MCS to sorted tuple of strings format"""
        standardized = []
        for mcs in mcs_list:
            # Convert to sorted tuple of strings
            if isinstance(mcs, (set, list)):
                # Handle both ['14'] and ['BE14'] formats
                standardized_items = []
                for item in mcs:
                    if isinstance(item, str) and item.startswith('BE'):
                        standardized_items.append(item[2:])  # 'BE14' -> '14'
                    else:
                        standardized_items.append(str(item))
                standardized.append(tuple(sorted(standardized_items)))
        return set(standardized)

    # Standardize both sets
    original_standardized = standardize_mcs(original_minimal_cut_sets)
    extracted_standardized = standardize_mcs(minimal_cut_sets_from_logs)

    print("\nOriginal Minimal Cut Sets (standardized):")
    for mcs in sorted(original_standardized):
        print(f"  {list(mcs)}")

    print("\nMinimal Cut Sets from generated logs (standardized):")
    for mcs in sorted(extracted_standardized):
        print(f"  {list(mcs)}")

    print(f"\nBoolean Expression from logs: TE = {boolean_expression_from_logs.replace('.', '·')}")

    # Compare with original
    print("\nComparison with original FT:")
    print(f"Original MCS count: {len(original_standardized)}")
    print(f"Generated MCS count: {len(extracted_standardized)}")
    print(f"MCS match: {original_standardized == extracted_standardized}")

    if original_standardized != extracted_standardized:
        print("\nDifferences (in standardized format):")
        missing = original_standardized - extracted_standardized
        extra = extracted_standardized - original_standardized

        if missing:
            print("Missing MCS in generated data:")
            for mcs in sorted(missing):
                print(f"  {list(mcs)}")

        if extra:
            print("Extra MCS in generated data:")
            for mcs in sorted(extra):
                print(f"  {list(mcs)}")

    return minimal_cut_sets_from_logs, boolean_expression_from_logs

print("\n" + "="*50)
print("SYNTHETIC FAULT LOG GENERATION WITH ACCELERATION")
print("="*50)

# FIX: Convert minimal cut sets to proper format for fault log generator
minimal_cut_sets_BE_formatted = []
for cut in minimal_cut_sets:
    formatted_cut = [f"BE{be}" for be in cut]
    minimal_cut_sets_BE_formatted.append(formatted_cut)

print("Formatted minimal cut sets for fault log generation:")
for mcs in minimal_cut_sets_BE_formatted:
    print(f"  {mcs}")

# Try different acceleration factors to get good coverage
acceleration_factors = [1000]
successful_generation = False
for acc_factor in acceleration_factors:
    print(f"\nTrying acceleration factor: {acc_factor}")
    try:
        fault_logs, truth_table_from_logs, accelerated_rates = add_fault_log_generation_to_main_accelerated(
            acceleration_factor=acc_factor
        )
        if (hasattr(truth_table_from_logs, 'shape') and
            len(truth_table_from_logs) > 5 and
            fault_logs['System_Failure'].sum() > 0):
            print(f"SUCCESS with acceleration factor {acc_factor}")
            print(f"System failures observed: {fault_logs['System_Failure'].sum()}")
            successful_generation = True
            break
        else:
            print(f"Acceleration factor {acc_factor} generated data but no system failures or small truth table")
    except Exception as e:
        print(f"Acceleration factor {acc_factor} failed: {e}")
        import traceback
        traceback.print_exc()
        continue

if not successful_generation:
    print("\nWARNING: No acceleration factor produced system failures!")
    print("Using the best available data...")

if hasattr(truth_table_from_logs, 'columns') and 'TE' in truth_table_from_logs.columns:
    minimal_cut_sets_from_logs, boolean_expression_from_logs = validate_and_compare_fault_trees(
        minimal_cut_sets, truth_table_from_logs
    )
else:
    print("\nERROR: truth_table_from_logs is not a valid truth table!")
    # Fallback: use original minimal cut sets
    minimal_cut_sets_from_logs = minimal_cut_sets
    boolean_expression_from_logs = boolean_expression

#%% Validate by reconstructing truth table from expression
tt_constructed = generate_truth_table_from_expression(boolean_expression)
tt_constructed.to_csv("output/truth_table_constructedFT.txt", sep=" ", index=False)
validate_truth_tables("output/truth_table_originalFT.txt", "output/truth_table_constructedFT.txt")

#%% Reliability Function (non-repairable mission, analytical)
# Uses extracted minimal_cut_sets and failure_rates to compute exact R(t).
import math
from collections import Counter

def _mcs_to_be_labels(min_cut_sets):
    mcs_be = []
    for cut in min_cut_sets:
        be_list = [f"BE{extract_event_number(be)}" for be in cut]
        mcs_be.append(tuple(sorted(be_list, key=lambda x: int(x[2:]))))
    return mcs_be

def _are_mcs_disjoint(mcs_be):
    counts = Counter([be for cut in mcs_be for be in cut])
    return max(counts.values(), default=0) <= 1

_lambda = dict(failure_rates)
mcs_be = _mcs_to_be_labels(minimal_cut_sets)
disjoint = _are_mcs_disjoint(mcs_be)
print(f"\n[Reliability] Minimal cut sets disjoint? {disjoint}")

def R_sys_disjoint(t_hours: float) -> float:
    prod_terms = 1.0
    for cut in mcs_be:
        Fi_prod = 1.0
        for be in cut:
            lam = _lambda[be]
            Fi_prod *= (1.0 - math.exp(-lam * t_hours))  # F_i(t)
        prod_terms *= (1.0 - Fi_prod)
    F_sys = 1.0 - prod_terms
    return 1.0 - F_sys

def _prob_TE_given_p(p_fail):
    from functools import lru_cache
    vars_all = sorted({be for cut in mcs_be for be in cut}, key=lambda x: int(x[2:]))
    cuts = tuple(frozenset(c) for c in mcs_be)

    @lru_cache(None)
    def _eval_prob(cuts_state, idx):
        for c in cuts_state:
            if len(c) == 0:
                return 1.0
        if idx == len(vars_all):
            return 0.0
        var = vars_all[idx]
        p1 = p_fail[var]
        p0 = 1.0 - p1
        cuts_if1 = tuple(frozenset(c - {var}) for c in cuts_state)
        cuts_if0 = tuple(frozenset(c) for c in cuts_state if var not in c)
        return p1 * _eval_prob(cuts_if1, idx + 1) + p0 * _eval_prob(cuts_if0, idx + 1)

    return _eval_prob(cuts, 0)

def R_sys_general(t_hours: float) -> float:
    p_fail = {be: 1.0 - math.exp(-_lambda[be] * t_hours) for be in _lambda}
    P_TE = _prob_TE_given_p(p_fail)
    return 1.0 - P_TE

R_sys = R_sys_disjoint if disjoint else R_sys_general

# Reliability curve R(t) System vs BE(s)
import matplotlib.ticker as ticker
T_cmp = 3.0e5  # hours
n_cmp = 1000   # smoothness
be_to_compare = ["BE11","BE10","BE15","BE5"]  # high contributors
time_cmp = np.linspace(0.0, T_cmp, n_cmp)
def R_be(lambda_per_hour, t_hours):
    return math.exp(-lambda_per_hour * t_hours)
R_sys_vals = np.array([R_sys(t) for t in time_cmp])

plt.figure(figsize=(10, 6), dpi=600)
plt.plot(time_cmp, R_sys_vals, color="black", linewidth=2, label="System (TE)")
for be in be_to_compare:
    lam = _lambda[be]
    plt.plot(time_cmp, np.array([R_be(lam, t) for t in time_cmp]),
             linestyle="--", label=be)
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Reliability", fontsize=14)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(4,4))
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# ============================
# 3) Map BEs -> Components
# ============================
be_to_component = {
    "BE1": "Fuse", "BE2": "Fuse",
    "BE3": "MCB",  "BE4": "MCB",
    "BE5": "PV Module", "BE6": "PV Module", "BE7": "PV Module", "BE8": "PV Module",
    "BE9": "PV Module", "BE10": "PV Module", "BE11": "PV Module", "BE12": "PV Module",
    "BE13": "PV Module", "BE14": "PV Module", "BE15": "PV Module", "BE16": "PV Module",
    "BE17": "Cable", "BE18": "Cable",
}

# ============================
# 4) Importance (Birnbaum, etc.)
# ============================
minimal_cut_sets_BE = [[f"BE{extract_event_number(be)}" for be in cut] for cut in minimal_cut_sets]

label_map = {f"BE{i}": i-1 for i in range(1, 19)}
failure_probs = {be: 1 - np.exp(-failure_rates[be] * T_mission) for be in failure_rates}
importance_df = calculate_importance_factors(minimal_cut_sets_BE, label_map, failure_probs)

print("\n--- Importance Factors (sample) ---")
print(importance_df)

#%% Simulation (DES)
# ============================
# 5) DES Simulation with memory measurement
# ============================
print("\n" + "="*50)
print("DISCRETE-EVENT SIMULATION (DES)")
print("="*50)

import tracemalloc

def measure_peak_memory(func, *args, **kwargs):
    """Run func with args and return (result, peak_memory_MB)."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1e6   # bytes -> MB

t_start = time.time()
res, des_peak_mem = measure_peak_memory(
    simulate_reliability,
    minimal_cut_sets=minimal_cut_sets_BE,
    failure_rates=failure_rates,
    repair_times=repair_times,
    T=T_mission,
    dt=1.0,
    N_SIM=N_SIM,
    be_to_component=be_to_component,
    rng_seed=123
)
t_des = time.time() - t_start
print(f"DES finished in {t_des:.2f} seconds, peak memory {des_peak_mem:.2f} MB")

time_grid = res["time_grid"]
sys_sim = res["system_stats"]
print("\nSystem metrics — simulation:")
print(pd.Series(sys_sim))

# --- System MTTF (analytical, NON-REPAIRABLE) from FT R_sys(t) ---
Tmax = 1.0e6
n_int = 4000
grid = np.linspace(0.0, Tmax, n_int)
R_vals = np.array([R_sys(t) for t in grid])
mtbf_trap = np.trapz(R_vals, grid)
lam_hat = fit_exponential_reliability(res["top_event_states"], dt=1.0, T=T_mission)
tail = (R_sys(Tmax) / lam_hat) if (lam_hat > 0) else 0.0
mttf_analytical_sys = mtbf_trap + tail
print("\nSystem MTTF (analytical, non-repairable):")
print(f"MTTF_analytical ≈ {mttf_analytical_sys:.6g} h")

# --- Analytic truncated MTTF from R_sys(t) ---
T = T_mission
grid = np.linspace(0.0, T, n_int)
R_vals = np.array([R_sys(t) for t in grid])
mttf_trunc_uncond_analytic = float(np.trapz(R_vals, grid))
p_fail_T = 1.0 - float(R_vals[-1])
mttf_trunc_cond_analytic = float(mttf_trunc_uncond_analytic / p_fail_T) if p_fail_T > 0 else float("inf")
print("\nSystem truncated MTTF (analytic, from R_sys):")
print(f"  Conditional  : {mttf_trunc_cond_analytic:.3f} h")
print(f"  Unconditional: {mttf_trunc_uncond_analytic:.3f} h")

# --- Component stats (MTTF, MTTR, λ, Unavailability) ---
comp = res["component_stats"]
print("\nComponent metrics (simulation-based):")
print(comp)

# Plot: bar chart of component unavailability (simulation)
plt.figure(figsize=(10,6))
comp_sorted = comp.sort_values("Unavailability", ascending=True)
plt.bar(comp_sorted.index, comp_sorted["Unavailability"].values, width=0.5)
plt.ylabel("Unavailability (fraction of time down)")
plt.title("Component Unavailability (simulation)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.xticks(fontsize=12, rotation=45, ha="right")
plt.yticks(fontsize=12)
plt.show()

# Log bar chart
plt.figure(figsize=(10,6), dpi=600)
comp_sorted = comp.sort_values("Unavailability", ascending=True)
plt.bar(comp_sorted.index, comp_sorted["Unavailability"].values, width=0.5)
plt.ylabel("Unavailability of the Component", fontsize=14)
plt.yscale("log")
plt.ylim(1e-7, 0.005)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=10))
ax.grid(True, which="major", axis="y", alpha=0.3)
ax.grid(False, which="minor", axis="y")
plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

print(f"\nSystem unavailability (mean over sims): {res['system_unavailability_mean']:.6g}")
print(f"95% CI for system unavailability: [{res['system_unavailability_ci'][0]:.6g}, {res['system_unavailability_ci'][1]:.6g}]\n")

# %% Cumulative Availability Plot (DES only)
down = res["top_event_states"]
cum_down_frac = down.cumsum(axis=1) / np.arange(1, down.shape[1]+1)
A_cum_sim = 1.0 - cum_down_frac
A_cum_mean = A_cum_sim.mean(axis=0)
A_cum_lo = np.percentile(A_cum_sim, 2.5, axis=0)
A_cum_hi = np.percentile(A_cum_sim, 97.5, axis=0)

plt.figure(figsize=(10,6), dpi=600)
plt.plot(time_grid, A_cum_mean, linewidth=1.5, label="Mean Cumulative Availability")
plt.fill_between(time_grid, A_cum_lo, A_cum_hi, alpha=0.18, label="Across-sim 95% band")
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Availability of the System", fontsize=14)
plt.title(f"Mean Cumulative Availability - {N_SIM} replications", fontsize=14)
plt.xlim(left=0, right=T_mission)
plt.ylim(top=1.00, bottom=0.98)
plt.grid(True, alpha=0.3)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(4,4))
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

#%%
# ============================
# 6) Proxel simulation with memory measurement
# ============================
print("\n" + "="*50)
print("PROXEL-BASED SIMULATION")
print("="*50)

# Define each basic event as a BasicEvent object (if not already defined)
belist = {}
for i in range(1, 19):
    be_name = f"BE{i}"
    lam = failure_rates[be_name]
    mu = 1.0 / repair_times[be_name]
    be = BasicEvent(
        states=['OK', 'F'],
        G=[[None, 1], [1, None]],
        dist=['exp', 'exp'],
        param=[(lam,), (mu,)]
    )
    belist[be_name] = be

# Parameters for proxel
dt_proxel = 1.0
tol = 1e-9

t_start = time.time()
prox_res, prox_peak_mem = measure_peak_memory(
    proxel_system,
    belist,
    minimal_cut_sets_BE,
    T_mission,
    dt_proxel,
    tol
)
t_prox = time.time() - t_start
print(f"Proxel finished in {t_prox:.2f} seconds, peak memory {prox_peak_mem:.2f} MB")

mean_unav_prox = np.trapz(prox_res['system_unavailability'], dx=dt_proxel) / T_mission
print(f"System mean unavailability (proxel): {mean_unav_prox:.6g}")
print(f"System mean unavailability (DES)   : {res['system_unavailability_mean']:.6g}")

# ============================
# 7) Comparison Metrics
# ============================
print("\n" + "="*50)
print("COMPARISON METRICS")
print("="*50)

# --- Reference probability (DES mean unavailability) ---
des_mean = res['system_unavailability_mean']
prox_mean = mean_unav_prox

# --- Relative error of proxel vs DES ---
rel_err_prox = abs(prox_mean - des_mean) / des_mean if des_mean > 0 else np.inf

# --- DES per‑simulation unavailability ---
down_des = res["top_event_states"]
per_sim_unav = down_des.mean(axis=1)          # fraction of time down per simulation
des_var = per_sim_unav.var(ddof=1) / N_SIM   # variance of the estimate
des_cv = np.sqrt(des_var) / des_mean if des_mean > 0 else np.inf

# Rare‑event hit count: number of simulations where system ever failed
ever_failed = down_des.any(axis=1)
hit_count = ever_failed.sum()
hit_rate = hit_count / N_SIM
cost_per_event = t_des / hit_count if hit_count > 0 else np.inf

# --- Array‑based memory (lower bound) ---
def array_memory(obj):
    if hasattr(obj, 'nbytes'):
        return obj.nbytes
    elif hasattr(obj, '__array__'):
        return np.array(obj).nbytes
    else:
        return 0

des_array_mem = down_des.nbytes / 1e6   # MB
be_unav = prox_res['be_unavailability']
be_mem = sum(arr.nbytes for arr in be_unav.values())
sys_mem = prox_res['system_unavailability'].nbytes
time_mem = prox_res['time_grid'].nbytes
prox_array_mem = (be_mem + sys_mem + time_mem) / 1e6   # MB

# --- MTTF/MTTR comparison ---
des_mttf = res['system_stats']['MTTF_sim']
des_mttr = res['system_stats']['MTTR_sim']
n_steps = len(prox_res['system_unavailability'])
steady_start = int(0.8 * n_steps)
U_ss = np.mean(prox_res['system_unavailability'][steady_start:])
mu_sys = 1.0 / des_mttr if des_mttr > 0 else 0.0
lambda_sys_prox = (U_ss * mu_sys) / (1.0 - U_ss) if U_ss < 1.0 else 0.0
mttf_prox = 1.0 / lambda_sys_prox if lambda_sys_prox > 0 else np.inf

# --- Print metrics table ---
print("\n{:<30} {:<18} {:<18}".format("Metric", "DES", "Proxel"))
print("-"*66)
print("{:<30} {:<18.2f} {:<18.2f}".format("Runtime (s)", t_des, t_prox))
print("{:<30} {:<18.2f} {:<18.2f}".format("Peak Memory (MB) (tracemalloc)", des_peak_mem, prox_peak_mem))
print("{:<30} {:<18.2f} {:<18.2f}".format("Array Memory (MB) (nbytes)", des_array_mem, prox_array_mem))
print("{:<30} {:<18.6f} {:<18.6f}".format("Mean Unavailability", des_mean, prox_mean))
print("{:<30} {:<18} {:<18.6f}".format("Relative Error", "reference", rel_err_prox))
# print("{:<30} {:<18.6f} {:<18}".format("CV (est.)", des_cv, "N/A"))
# print("{:<30} {:<18.0f} {:<18}".format("Hit count", hit_count, "N/A"))
# print("{:<30} {:<18.6f} {:<18}".format("Cost per event (s)", cost_per_event, "N/A"))
# print("{:<30} {:<18.4f} {:<18}".format("Hit rate", hit_rate, "N/A"))
print("{:<30} {:<18.2f} {:<18.2f}".format("MTTF (h)", des_mttf, mttf_prox))
# print("{:<30} {:<18.2f} {:<18}".format("MTTR (h)", des_mttr, "N/A"))

#%%

# ============================
# 8) Cumulative Availability Comparison
# ============================
print("\n" + "="*50)
print("CUMULATIVE AVAILABILITY COMPARISON")
print("="*50)

# DES cumulative availability (mean across replications)
down_des = res["top_event_states"]               # shape (N_SIM, Nt)
N_sim, Nt = down_des.shape
time_grid_des = res["time_grid"]                  # length Nt

# Fraction of time the system has been down up to each time step
cum_down_frac = down_des.cumsum(axis=1) / np.arange(1, Nt+1).reshape(1, -1)
A_cum_sim = 1.0 - cum_down_frac                   # cumulative availability per replication
A_cum_mean_des = A_cum_sim.mean(axis=0)           # mean across replications

# Compute 95% confidence interval (normal approximation)
A_cum_std = A_cum_sim.std(axis=0, ddof=1)         # sample standard deviation
A_cum_se = A_cum_std / np.sqrt(N_sim)              # standard error
margin = 1.96 * A_cum_se                           # half-width of 95% CI
ci_lower = A_cum_mean_des - margin
ci_upper = A_cum_mean_des + margin

# Proxel cumulative availability
avail_prox = 1.0 - np.array(prox_res['system_unavailability'])
cum_avail_prox = np.cumsum(avail_prox) / (np.arange(1, len(avail_prox)+1))

# Create the plot
plt.figure(figsize=(10,6), dpi=600)
plt.plot(time_grid_des, A_cum_mean_des, linewidth=2, label='DES (mean cumulative availability)')
plt.fill_between(time_grid_des, ci_lower, ci_upper, alpha=0.2, label='95% CI (DES)')
plt.plot(prox_res['time_grid'], cum_avail_prox, '--', linewidth=2, label='Proxel (cumulative availability)')
plt.xlabel('Time (hours)', fontsize=14)
plt.ylabel('Cumulative Availability', fontsize=14)
plt.title('Proxel vs DES: Cumulative System Availability')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.ylim(0.995, 1.00)
plt.xlim(0, 10000)
plt.tight_layout()
plt.savefig('output/cumulative_availability_comparison.png', dpi=600)
plt.show()

#%%

# ============================
# 9) Exports & Reliability Function
# ============================
import csv
out_csv = "time_A_R.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_hours", "Availability_A", "Reliability_R"])
    for t, a, r in zip(time_grid, res["availability_time_series"], res["reliability_time_series"]):
        w.writerow([float(t), float(a), float(r)])
print(f"\nSaved curves to {out_csv}")

R_of_t = make_reliability_function(time_grid, res["reliability_time_series"])
for t_demo in [0, 50, 100, 250, 500, 1000]:
    print(f"R({t_demo} h) ≈ {R_of_t(t_demo):.6f}")

lam_hat = fit_exponential_reliability(res["top_event_states"], dt=1.0, T=T_mission)
print(f"\nExponential fit λ̂ = {lam_hat:.6g}  =>  Ĥ R(t) = exp(-λ̂ t)")
for t_demo in [100, 250, 500, 1000]:
    print(f"R_exp_hat({t_demo} h) = {np.exp(-lam_hat * t_demo):.6f}")

# Keep plots open
plt.show()