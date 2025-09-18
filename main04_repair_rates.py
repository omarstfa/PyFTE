
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Components ---
from components.event import Event, extract_event_number
from components.gate import Gate
from components.fault_tree import FaultTree, print_fault_tree
from components.cutset import extract_cut_sets, get_minimal_cut_sets, build_boolean_expression
from components.truth_table import generate_truth_table_from_expression
from components.validate import validate_truth_tables
from components.critical_rank import calculate_importance_factors, simulate_reliability,make_reliability_function, fit_exponential_reliability

#%%

# ============================
# 1) Build Fault Tree (FT)
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
tt_simple.to_csv("truth_table_originalFT.txt", sep=" ", index=False)

# --- Minimal cut sets & Boolean expression ---
cut_sets = extract_cut_sets(tt_simple)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)
boolean_expression = build_boolean_expression(minimal_cut_sets)

print("\nMinimal Cut Sets:")
for mcs in minimal_cut_sets:
    print(sorted(mcs))

print("\nExtracted Fault Tree Boolean Expression: TE =", boolean_expression.replace('.', '·'))

# Validate by reconstructing truth table from expression
tt_constructed = generate_truth_table_from_expression(boolean_expression)
tt_constructed.to_csv("truth_table_constructedFT.txt", sep=" ", index=False)
validate_truth_tables("truth_table_originalFT.txt", "truth_table_constructedFT.txt")

# ============================
# 2) Inputs (rates, repairs)
# ============================
T_mission = 1000  # hours
N_SIM = 200 # replications

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

# Switch to REPAIR RATES (mu per hour). If you formerly used MTTR (hours),
# set mu = 1 / MTTR. Below mirrors the previously active repair_times dict.
repair_rates = {
    'BE1': 1/4,   'BE2': 1/4,
    'BE3': 1/10,  'BE4': 1/10,
    'BE5': 1/24,  'BE6': 1/24,
    'BE7': 1/24,  'BE8': 1/80,
    'BE9': 1/80,  'BE10': 1/48,
    'BE11': 1/48, 'BE12': 1/24,
    'BE13': 1/24, 'BE14': 1/12,
    'BE15': 1/48, 'BE16': 1/48,
    'BE17': 1/4,  'BE18': 1/4,
}

def scale_repair_rates(rates: dict, factor: float) -> dict:
    """Return a new dict with all repair rates multiplied by factor."""
    return {k: v * factor for k, v in rates.items()}

# Example: divide all rates by 2
repair_rates = scale_repair_rates(repair_rates, 0.5)

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
T_cmp = 2.0e5      # hours
n_cmp = 5000
time_cmp = np.linspace(0.0, T_cmp, n_cmp)

def R_be(lambda_per_hour, t_hours):
    return math.exp(-lambda_per_hour * t_hours)

# pick a concise, informative set (edit as desired)
be_to_compare = ["BE11","BE10","BE15","BE5"]  # high contributors
# be_to_compare = ["BE1","BE2","BE3","BE4"]  # Least critical
R_sys_vals = np.array([R_sys(t) for t in time_cmp])

plt.figure(figsize=(10, 6))
plt.plot(time_cmp / 1000.0, R_sys_vals, linewidth=2, label="System (TE)")
for be in be_to_compare:
    lam = _lambda[be]
    plt.plot(time_cmp / 1000.0, np.array([R_be(lam, t) for t in time_cmp]), linestyle="--", label=be)

plt.xlabel("Time (hours)")
plt.ylabel("Reliability")
plt.title("Comparison: System vs BE Reliability")
plt.grid(True, alpha=0.3)
plt.legend()
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
# %%

T_mission = 10000  # hours
N_SIM = 1500 # replications

# ============================
# 5) Availability & Reliability (Monte Carlo)
# ============================
res = simulate_reliability(
    minimal_cut_sets=minimal_cut_sets_BE,
    failure_rates=failure_rates,
    repair_rates=repair_rates,
    T=T_mission,
    dt=1.0,
    N_SIM=N_SIM,
    be_to_component=be_to_component,
    rng_seed=123,
)

time_grid = res["time_grid"]

# --- Print concise summary ---
print(f"\nSystem unavailability (mean over sims): {res['system_unavailability_mean']:.6g}")
print(f"95% CI for system unavailability: [{res['system_unavailability_ci'][0]:.6g}, {res['system_unavailability_ci'][1]:.6g}]\n")

print("Component unavailability (mean):")
print(res["component_unavailability"].sort_values("Unavailability", ascending=False))

# ============================
# 6) Plots
# ============================
plt.figure(figsize=(10,6))
plt.plot(time_grid, res["availability_time_series"], label="A(t)")
plt.fill_between(time_grid, res["availability_time_low"], res["availability_time_high"], alpha=0.2, label="95% CI")
plt.xlabel("Time (hours)")
plt.ylabel("Availability")
plt.title("System Availability A(t) with 95% CI")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(time_grid, res["reliability_time_series"], label="R(t)")
plt.fill_between(time_grid, res["reliability_time_low"], res["reliability_time_high"], alpha=0.2, label="95% CI")
plt.xlabel("Time (hours)")
plt.ylabel("Reliability")
plt.title("System Reliability R(t) with 95% CI")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================
# Cumulative Availability Plot
# ============================
# A_cum(t) = running mean of system up-state since time 0
# Computed per Monte Carlo replication, then averaged across replications.

down = res["top_event_states"]  # shape: (N_SIM, Nt), True when system is DOWN

# Per replication: running mean of down states → cumulative availability
cum_down_frac = down.cumsum(axis=1) / np.arange(1, down.shape[1]+1)  # fraction down up to time t
A_cum_sim = 1.0 - cum_down_frac                                     # per-replication cumulative availability

# Mean and 95% band across replications
A_cum_mean = A_cum_sim.mean(axis=0)
A_cum_lo = np.percentile(A_cum_sim, 2.5, axis=0)
A_cum_hi = np.percentile(A_cum_sim, 97.5, axis=0)

plt.figure(figsize=(10,6))
# for j in range(50):  # first 5 replications
#     plt.plot(time_grid, A_cum_sim[j], alpha=0.5, linewidth=1)
plt.plot(time_grid, A_cum_mean, color="red", linewidth=2, label="Mean Cumulative Availability")
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative Availability")
plt.title(f"Mean Cumulative Availability - {N_SIM} replications")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# ============================
# Cumulative Availability Plot + 95% band (zoomed on mean) 
# ============================
# Inputs expected:
#   time_grid: 1D array shape (Nt,), strictly increasing
#   res["top_event_states"]: bool array shape (N_SIM, Nt) OR (Nt, N_SIM)
#                            True when the SYSTEM is DOWN

down = res["top_event_states"]
Nt = len(time_grid)

plt.figure(figsize=(10, 6))
plt.plot(time_grid, A_cum_mean, lw=2, label="Mean cumulative availability")
plt.fill_between(time_grid, A_cum_lo, A_cum_hi, alpha=0.18, label="95% band")
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative Availability")
plt.title("Cumulative Availability — Mean and 95% Band (zoomed)")
plt.ylim(0.996, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%%

# ============================
# 7) Export & Reliability Function
# ============================
import csv

out_csv = "time_A_R.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_hours", "Availability_A", "Reliability_R"])
    for t, a, r in zip(time_grid, res["availability_time_series"], res["reliability_time_series"]):
        w.writerow([float(t), float(a), float(r)])
print(f"\nSaved curves to {out_csv}")

# Build an interpolated Reliability Function R(t)
R_of_t = make_reliability_function(time_grid, res["reliability_time_series"])
for t_demo in [0, 50, 100, 250, 500, 1000]:
    print(f"R({t_demo} h) ≈ {R_of_t(t_demo):.6f}")

# Fit exponential reliability (with censoring) for a compact closed-form approx
lam_hat = fit_exponential_reliability(res["top_event_states"], dt=1.0, T=T_mission)
print(f"\nExponential fit λ̂ = {lam_hat:.6g}  =>  Ĥ R(t) = exp(-λ̂ t)")
for t_demo in [100, 250, 500, 1000]:
    print(f"R_exp_hat({t_demo} h) = {np.exp(-lam_hat * t_demo):.6f}")
