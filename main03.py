import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from components.event import Event
from components.event import extract_event_number
from components.gate import Gate
from components.fault_tree import FaultTree
from components.fault_tree import print_fault_tree
from components.cutset import extract_cut_sets, get_minimal_cut_sets, build_boolean_expression
from components.boolean_parser import parse_expression, print_tree
from components.truth_table import generate_truth_table_from_expression
from components.validate import validate_truth_tables
from components.critical_rank import calculate_importance_factors, simulate_reliability


# === Original Fault Tree Structure ===
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

tree_original = FaultTree(topEvent)
truth_table_original = tree_original.generate_truth_table()
truth_table_original_simple = truth_table_original.copy()
column_mapping = {name: extract_event_number(name) for name in truth_table_original_simple.columns}
column_mapping["Top Event"] = "TE"
truth_table_original_simple.rename(columns=column_mapping, inplace=True)
sorted_cols = sorted([col for col in truth_table_original_simple.columns if col != "TE"], key=lambda x: int(x)) + ["TE"]
truth_table_original_simple = truth_table_original_simple[sorted_cols]
truth_table_original_simple.to_csv("truth_table_originalFT.txt", sep=' ', index=False)

print("\nOriginal Truth Table (sample):")
truth_table_original_BE = truth_table_original_simple.copy()
truth_table_original_BE.columns = [f"BE{col}" if col != 'TE' else 'TE' for col in truth_table_original_BE.columns]
print(truth_table_original_BE.head())

# === Cut Sets & Boolean Expression ===
cut_sets = extract_cut_sets(truth_table_original_simple)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)
boolean_expression = build_boolean_expression(minimal_cut_sets)

print("\nMinimal Cut Sets:")
for mcs in minimal_cut_sets:
    print(sorted(mcs))

print("\nExtracted Fault Tree Boolean Expression: TE =", boolean_expression.replace('.', '·'))

# === Construct Tree from Boolean Expression ===
print("\nConstructed Fault Tree Structure:")
root = parse_expression(boolean_expression)
print_tree(root)

# === Generated Truth Table from Boolean Expression ===
truth_table_constructed = generate_truth_table_from_expression(boolean_expression)
truth_table_constructed_BE = truth_table_constructed.copy()
truth_table_constructed_BE.columns = [f"BE{col}" if col != 'TE' else 'TE' for col in truth_table_constructed_BE.columns]
print("\nConstructed Truth Table (sample):")
print(truth_table_constructed_BE.head())

truth_table_constructed.to_csv("truth_table_constructedFT.txt", sep=" ", index=False)

# === Load and Validate Truth Tables ===
validate_truth_tables("truth_table_originalFT.txt", "truth_table_constructedFT.txt")


#%% Analysis: Importance & Unavailability

T_mission = 1000  # time units
N_SIM = 100       # number of Monte Carlo simulations

# === Mapping and probabilities ===
label_map = {str(i): f'BE{i}' for i in range(1, 19)}

failure_rates = {
                'BE1':1e-7, 'BE2':2e-7,
                'BE3':8e-7, 'BE4':8e-7,
                'BE5':8.46e-6,  'BE6':4.9e-6,
                'BE7':3e-7, 'BE8':1.3e-6,
                'BE9':8.8e-6, 'BE10':1.115e-5,
                'BE11':1.487e-5,'BE12':1.01e-6,
                'BE13':2.1e-7,'BE14':5.2e-7,
                'BE15': 7.29e-6, 'BE16':5.7e-6,
                'BE17':1e-7,'BE18':2e-7
                }
repair_times = {
                'BE1': 3,  'BE2': 3,
                'BE3': 6,  'BE4': 6,
                'BE5': 12, 'BE6': 12,
                'BE7': 6,  'BE8': 6,
                'BE9': 6,  'BE10': 12,
                'BE11': 12,'BE12': 6,
                'BE13': 3, 'BE14': 3,
                'BE15': 12,'BE16': 12,
                'BE17': 3, 'BE18': 3
            }
# repair_times = {
#                 'BE1': 30,  'BE2': 30,
#                 'BE3': 60,  'BE4': 60,
#                 'BE5': 120, 'BE6': 120,
#                 'BE7': 60,  'BE8': 60,
#                 'BE9': 60,  'BE10': 120,
#                 'BE11': 120,'BE12': 60,
#                 'BE13': 30, 'BE14': 30,
#                 'BE15': 120,'BE16': 120,
#                 'BE17': 30, 'BE18': 30
#             }
# Convert set-style cut sets into lists of numbers (e.g., '1' -> 'BE1')
mcs_numeric = [[extract_event_number(be) for be in cut] for cut in minimal_cut_sets]

# === Compute Importance Factors ===

# Convert failure rates to failure probabilities for importance
failure_probs = {be: 1 - np.exp(-rate * T_mission) for be, rate in failure_rates.items()}

importance_df = calculate_importance_factors(mcs_numeric, label_map, failure_probs)
print("\n--- Importance Factors ---")
print(importance_df)

#%% Reliability Function

# === Reliability function from extracted Fault Tree (non-repairable mission) ===
# Place this block after you build 'minimal_cut_sets' and 'boolean_expression'

import math
from collections import Counter, defaultdict

# Helper: turn "['3','4']" style cut sets into canonical BE labels like 'BE3'
def _mcs_to_be_labels(min_cut_sets):
    mcs_be = []
    for cut in min_cut_sets:
        be_list = [f"BE{extract_event_number(be)}" for be in cut]
        mcs_be.append(tuple(sorted(be_list, key=lambda x: int(x[2:]))))
    return mcs_be

# Helper: check if minimal cut sets are disjoint (no BE appears in two different cuts)
def _are_mcs_disjoint(mcs_be):
    counts = Counter([be for cut in mcs_be for be in cut])
    return max(counts.values(), default=0) <= 1

# Build a fast lookup for lambdas per BE
_lambda = {be: failure_rates[be] for be in failure_rates.keys()}

mcs_be = _mcs_to_be_labels(minimal_cut_sets)
disjoint = _are_mcs_disjoint(mcs_be)

print(f"\n[Reliability] Minimal cut sets disjoint? {disjoint}")

# --- Exact closed form when MCS are disjoint ---
def R_sys_disjoint(t_hours: float) -> float:
    # F_sys(t) = 1 - Π_k (1 - Π_{i in cut_k} F_i(t));  R = 1 - F_sys
    prod_terms = 1.0
    for cut in mcs_be:
        Fi_prod = 1.0
        for be in cut:
            lam = _lambda[be]
            Fi_prod *= (1.0 - math.exp(-lam * t_hours))  # F_i(t)
        prod_terms *= (1.0 - Fi_prod)
    F_sys = 1.0 - prod_terms
    return 1.0 - F_sys

# --- General exact evaluator via Shannon decomposition (handles overlapping MCS) ---
# We evaluate the probability of the Boolean top event from the truth-function defined by the minimal cut sets:
# TE = OR over cuts (AND over their BEs). This is exact and avoids double counting.
def _prob_TE_given_p(p):
    """
    p: dict { 'BE1': F1(t), ... } = failure probabilities at time t
    Returns P(TE=1) exactly using Shannon expansion over variables.
    """
    # Build CNF-like function f(x) = OR_k AND_{i in cut_k} x_i, where x_i ~ Bernoulli(p_i), independent.
    # Memoized recursion on a variable ordering.
    from functools import lru_cache

    # Collect variables
    vars_all = sorted({be for cut in mcs_be for be in cut}, key=lambda x: int(x[2:]))

    # Represent each cut as frozenset of variables
    cuts = tuple(frozenset(c) for c in mcs_be)

    @lru_cache(None)
    def _eval_prob(cuts_state, idx):
        # cuts_state: tuple of frozensets (remaining literals per cut)
        # idx: next variable index in vars_all to branch on
        # If any cut is empty -> already satisfied (AND of zero literals = True), so TE=1 w.p. 1
        for c in cuts_state:
            if len(c) == 0:
                return 1.0
        # If no cuts left (shouldn't happen) or no variables left, TE cannot occur
        if idx == len(vars_all):
            return 0.0

        var = vars_all[idx]
        p1 = p[var]       # P(var=1) = failure prob of that BE at time t
        p0 = 1.0 - p1

        # Branch var=1: remove var from each cut (because it's satisfied for those cuts that need it)
        cuts_if1 = tuple(frozenset(c - {var}) for c in cuts_state)
        # Branch var=0: for any cut that needs var, that cut becomes impossible (since AND requires it)
        cuts_if0 = tuple(frozenset(c) for c in cuts_state if var not in c)

        return p1 * _eval_prob(cuts_if1, idx + 1) + p0 * _eval_prob(cuts_if0, idx + 1)

    return _eval_prob(cuts, 0)

def R_sys_general(t_hours: float) -> float:
    # Build per-BE failure probabilities at time t, Fi(t) = 1 - exp(-λ_i t)
    p_fail = {be: 1.0 - math.exp(-_lambda[be] * t_hours) for be in _lambda}
    P_TE = _prob_TE_given_p(p_fail)
    return 1.0 - P_TE

# Choose which evaluator to use based on disjointness
R_sys = R_sys_disjoint if disjoint else R_sys_general


# Reliability Plot

# Choose time horizon for plotting (hours)
T_plot = 1.0e5    # 100,000 hours
n_points = 5000    # resolution

# Build time grid (in hours)
time_grid_R = np.linspace(0, T_plot, n_points)

# Evaluate reliability
R_vals = np.array([R_sys(t) for t in time_grid_R])

# Plot, dividing time axis by 1000
plt.figure(figsize=(10, 6))
plt.plot(time_grid_R / 1000.0, R_vals, linewidth=2)

plt.xlabel("Time (×1000 hours)", fontsize=14)
plt.ylabel("Reliability R(t)", fontsize=14)
plt.title("System Reliability Function", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# Comparision PLot

# ---- Settings ----
T_plot = 500e3        # total time horizon in hours (e.g., 100,000 h)
n_points = 600        # curve resolution
be_to_compare = ["BE11","BE15","BE6","BE4"]  # add any BEs you want to compare here

# ---- Time grid (hours) ----
time_grid = np.linspace(0.0, T_plot, n_points)

# ---- System reliability values (assumes R_sys(t) is already defined) ----
R_sys_vals = np.array([R_sys(t) for t in time_grid])

# ---- Basic-event reliability helper ----
def R_be(lambda_per_hour, t_hours):
    # Exponential reliability: R(t) = exp(-λ t)
    return math.exp(-lambda_per_hour * t_hours)

# ---- Compute chosen BE curves ----
be_curves = {}
for be in be_to_compare:
    if be not in failure_rates:
        raise KeyError(f"{be} not found in failure_rates")
    lam = failure_rates[be]
    be_curves[be] = np.array([R_be(lam, t) for t in time_grid])

# ---- Plot (time axis divided by 1000) ----
plt.figure(figsize=(10, 6))
plt.plot(time_grid / 1000.0, R_sys_vals, linewidth=2, label="System R(t)")

for be, vals in be_curves.items():
    plt.plot(time_grid / 1000.0, vals, linestyle="--", label=f"{be} R(t)")

plt.xlabel("Time (×1000 hours)", fontsize=13)
plt.ylabel("Reliability R(t)", fontsize=13)
plt.title("System vs. Basic-Event Reliability", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%% Reliability and Availability Simulation 2

import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
rng = np.random.default_rng(42)

# ---------- Fault logic (adapt to your extracted tree if needed) ----------
# System is DOWN if any singleton BE fails OR (BE3 AND BE4) both failed.
singletons = {"BE1","BE2","BE5","BE6","BE7","BE8","BE9","BE10","BE11","BE12",
              "BE13","BE14","BE15","BE16","BE17","BE18"}
pair_and = ("BE3","BE4")

def system_down(component_states: dict) -> bool:
    # component_states[BE] is True if failed, False if up
    if any(component_states[be] for be in singletons):
        return True
    return component_states[pair_and[0]] and component_states[pair_and[1]]

# ---------- 1) Mission reliability simulation (NO repairs) ----------
# Trick: time to system failure T_sys for this tree = min( min(T_i for singletons), max(T3, T4) ),
# where T_i ~ Exp(lambda_i). We can simulate that directly.
def simulate_mission_reliability(T_plot=1e5, n_points=400, n_trials=100_000):
    time_grid = np.linspace(0, T_plot, n_points)
    # Precompute lambdas
    lam = failure_rates
    # Draw samples of time-to-failure per BE for all trials
    # (exponential inverse CDF: T = -ln(U)/lambda)
    def exp_samples(rate, size):
        # Guard for zero rates (not expected here)
        if rate <= 0:
            return np.full(size, np.inf)
        U = rng.random(size)
        return -np.log(U) / rate

    # Singles: min over 16 BEs
    singles_min = np.full(n_trials, np.inf, dtype=float)
    for be in singletons:
        singles_min = np.minimum(singles_min, exp_samples(lam[be], n_trials))

    # AND pair needs both failures: time is max(T3, T4)
    T3 = exp_samples(lam["BE3"], n_trials)
    T4 = exp_samples(lam["BE4"], n_trials)
    pair_time = np.maximum(T3, T4)

    T_sys = np.minimum(singles_min, pair_time)

    # Reliability estimator: R(t) = P(T_sys > t) ≈ mean( T_sys > t )
    R_vals = np.array([(T_sys > t).mean() for t in time_grid])
    return time_grid, R_vals

# ---------- 2) Availability simulation (WITH repairs) ----------
# Event-driven simulation with unlimited repair crews.
# Each component alternates: Up --(Exp(lambda))-> Down --(fixed repair)-> Up ...
def simulate_availability(T_horizon=1e5, n_points=400, n_trials=2000):
    time_grid = np.linspace(0, T_horizon, n_points)
    A_accum = np.zeros_like(time_grid, dtype=float)

    be_list = sorted(failure_rates.keys(), key=lambda x: int(x[2:]))
    lam = failure_rates
    rep = repair_times

    for _ in range(n_trials):
        # Initial states: all up (False=up, True=failed)
        state = {be: False for be in be_list}

        # Priority queue of events: (time, type, be)
        # type: "fail" or "repair"
        Q = []
        for be in be_list:
            # schedule initial failure when up
            t_fail = rng.exponential(1.0 / lam[be]) if lam[be] > 0 else np.inf
            heapq.heappush(Q, (t_fail, "fail", be))

        # We'll walk through the event list, and sample the state at each time_grid point
        idx = 0
        t_now = 0.0

        # Helper to advance to a target time and ensure A(t) sampling at grid points
        def sample_until(target_time):
            nonlocal idx, t_now
            while idx < len(time_grid) and time_grid[idx] <= target_time:
                # sample instantaneous availability: system is up if not system_down
                A_accum[idx] += 0.0 if system_down(state) else 1.0
                idx += 1
            t_now = target_time

        while Q and idx < len(time_grid):
            t_evt, etype, be = heapq.heappop(Q)
            if t_evt > time_grid[-1]:
                # sample the remainder and break
                sample_until(time_grid[-1])
                break

            # sample at all grid points up to the next event
            sample_until(t_evt)

            if etype == "fail":
                if not state[be]:  # only fail if currently up
                    state[be] = True
                    # schedule fixed repair
                    t_rep = t_evt + rep[be]
                    heapq.heappush(Q, (t_rep, "repair", be))
                # schedule next failure for this BE after it returns to up.
                # We schedule the next failure time when it returns to up (on repair).
                # (So nothing else to do here.)
            elif etype == "repair":
                if state[be]:  # only repair if currently failed
                    state[be] = False
                    # schedule next failure from this repair completion time
                    if lam[be] > 0:
                        t_next_fail = t_evt + rng.exponential(1.0 / lam[be])
                    else:
                        t_next_fail = np.inf
                    heapq.heappush(Q, (t_next_fail, "fail", be))

        # If grid extends beyond last event time, sample the rest at last known state
        if idx < len(time_grid):
            sample_until(time_grid[-1])

    # Empirical instantaneous availability A(t) = mean over trials of "system up" at t
    A_vals = A_accum / n_trials
    return time_grid, A_vals

# ---------- RUN SIMS ----------
T_plot = 1.0e5
n_points = 500

# Analytic reliability (already from your R_sys)
t_analytic = np.linspace(0, T_plot, n_points)
R_analytic = np.array([R_sys(t) for t in t_analytic])

# Mission reliability (no repairs) Monte Carlo
t_Rmc, R_mc = simulate_mission_reliability(T_plot=T_plot, n_points=n_points, n_trials=100000)

# Availability with repairs Monte Carlo
t_Amc, A_mc = simulate_availability(T_horizon=T_plot, n_points=n_points, n_trials=3000)

# ---------- PLOTS (time divided by 1000) ----------
plt.figure(figsize=(10, 6))
plt.plot(t_analytic/1000.0, R_analytic, label="Analytic R(t) (no repairs)", linewidth=2)
plt.plot(t_Rmc/1000.0, R_mc, "--", label="Simulated R(t) (no repairs)", alpha=0.8)
plt.xlabel("Time (×1000 hours)")
plt.ylabel("Reliability R(t)")
plt.title("Mission Reliability: Analytic vs Monte Carlo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_Amc/1000.0, A_mc, label="Simulated Availability A(t) (with repairs)", linewidth=2)
plt.xlabel("Time (×1000 hours)")
plt.ylabel("Availability A(t)")
plt.title("Time-Series Availability with Failures and Fixed Repairs")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%% Reliability and Availability Simulation

# === Simulate Reliability ===
minimal_cut_sets_BE = [[f'BE{extract_event_number(be)}' for be in cut] for cut in minimal_cut_sets]

(
    sys_unavail,
    sys_ci,                   
    comp_df,
    availability_time_series,
    time_grid,
    all_component_states,
    availability_sys,
    MTTF_sys,
    MTTR_sys,
    comp_unavail_summary,     
    sys_unavail_ci,            
    availability_time_low,
    availability_time_high,
    top_event_states,
    reliability_time_series,
    reliability_time_low,
    reliability_time_high,
    availability_mission,
    reliability_mission,
                            ) = simulate_reliability(
                                                    minimal_cut_sets_BE,
                                                    failure_rates,
                                                    repair_times,
                                                    T=T_mission,
                                                    dt=1.0
                                                    )

#%% Results
print(f"\nSystem Unavailability (mean of {N_SIM} sims over T={T_mission}h): {sys_unavail:.6g}")
print(f"95% CI of Unavailability: [{sys_unavail_ci[0]:.6g}, {sys_unavail_ci[1]:.6g}]")

print(f"\nEmpirical Availability (MTTF/(MTTF+MTTR)): {availability_sys}")
print(f"MTTF ≈ {MTTF_sys}   |   MTTR ≈ {MTTR_sys}")

# # Components: point estimate + CI (show first few)
# print("\nComponent unavailability (mean):")
# print(comp_df[['Unavailability']])

# print("\nComponent unavailability (95% CI):")
# print(comp_unavail_summary)

#%% Sorting Components by Unavailability
def _sort_by_event_number(df):
    return df.sort_index(key=lambda s: s.str.extract(r'(\d+)').astype(int)[0])

importance_df_sorted = _sort_by_event_number(importance_df)
comp_df_sorted = _sort_by_event_number(comp_df)
comp_ci_sorted = _sort_by_event_number(comp_unavail_summary)

# Scale component unavailability for readability to 1e-3 hrs (per 1000 hrs)
comp_df_sorted = comp_df_sorted.copy()
comp_df_sorted["Unavailability (1e-3)"] = comp_df_sorted["Unavailability"] * 1e3

print("\nComponents sorted by unavailability (mean):")
print(comp_df_sorted.sort_values("Unavailability", ascending=False))

# print("\nComponent unavailability with 95% CI:")
# print(comp_ci_sorted)

#%% PLots

# Plot Unavailability of Basic Events 
plt.figure(figsize=(10, 6))
ax = plt.gca()  # Get current Axes
ax.bar(comp_df.index, comp_df['Unavailability']/1000, label='Basic Events')
# ax.bar('Top Event', sys_unavail/1000, color='orange', label='Top Event')
ax.set_ylabel('Unavailability', fontsize=14)
ax.tick_params(axis='x', rotation=45, labelsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.yaxis.get_offset_text().set_fontsize(12)  # Set offset text font size
# ax.legend()
plt.tight_layout()
plt.show()


# Plot Time-Series Availability of Top Event
plt.figure(figsize=(10, 6))
plt.plot(time_grid, availability_time_series, label='System Availability')
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Availability", fontsize=14)
# plt.title("System Availability Over Time", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
# plt.legend()
plt.show()

#%% System Availability CI 95%

plt.figure(figsize=(10, 6))
plt.plot(time_grid, availability_time_series, label="Mean Availability")
plt.fill_between(time_grid, availability_time_low, availability_time_high, alpha=0.2, label="95% Confidence Interval")
plt.ylim(0.995, 1.0005)
plt.xlabel("Time (hours)")
plt.ylabel("System availability")
plt.title("System Availability with 95% Confidence Interval")
plt.legend()
plt.tight_layout()
plt.show()

#%% System R(t)

plt.figure(figsize=(10, 6))
plt.plot(time_grid, reliability_time_series, label="Reliability R(t)")
plt.fill_between(time_grid, reliability_time_low, reliability_time_high, alpha=0.2, label="95% CI (Wilson)")
plt.xlabel("Time (hours)")
plt.ylabel("R(t)")
plt.title("System Reliability R(t) with 95% CI")
# plt.ylim(0.95, 1.001)
plt.legend()
plt.tight_layout()
plt.show()

#%%

plt.figure(figsize=(10, 6))
plt.plot(availability_mission, label="Availability A(t)")

plt.xlabel("Time (hours)")
plt.ylabel("A(t)")

plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(reliability_mission, label="Reliability R(t)")

plt.xlabel("Time (hours)")
plt.ylabel("R(t)")

plt.legend()
plt.tight_layout()
plt.show()


#%% Exporting Results Data

# Sort helper function
def sort_by_event_number(df):
    return df.sort_index(key=lambda x: x.str.extract(r'(\d+)').astype(int)[0])

# Sort importance and reliability DataFrames
importance_df_sorted = sort_by_event_number(importance_df)
comp_df_sorted = sort_by_event_number(comp_df)

comp_df_sorted.rename(columns={"Unavailability": "Unavailability (1e-3)"})
comp_df_sorted['Unavailability'] = comp_df_sorted['Unavailability']*1000


# # Export to CSV
# importance_df_sorted.to_csv("importance_factors_sorted.csv")
# comp_df_sorted.to_csv("reliability_metrics_sorted.csv")
# print("Exported sorted importance and reliability data to CSV.")
