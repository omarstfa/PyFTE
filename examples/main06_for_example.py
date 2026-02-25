
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Components ---
from components.event import Event, extract_event_number
from components.gate import Gate
from components.fault_tree import FaultTree, print_fault_tree
from components.cutset import extract_cut_sets, get_minimal_cut_sets, build_boolean_expression
from components.truth_table import generate_truth_table_from_expression
from components.fault_log_generator_example import generate_synthetic_fault_logs_accelerated, fault_logs_to_truth_table
from components.validate import validate_truth_tables
from components.critical_rank import calculate_importance_factors, simulate_reliability,make_reliability_function, fit_exponential_reliability

#%%

# ============================
# 1) Build A Fault Tree (FT)
# ============================

topEvent = Event('Top Event')
or_gate = Gate('OR', parent=topEvent)
# Basic events directly under OR gate
Event('Basic Event 1', parent=or_gate)
Event('Basic Event 2', parent=or_gate)
# One AND gate for demonstration
and_gate = Gate('AND', parent=or_gate)
Event('Basic Event 3', parent=and_gate) 
Event('Basic Event 4', parent=and_gate)


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
# tt_simple.to_csv("output/truth_table_originalFT.txt", sep=" ", index=False)

# --- Minimal cut sets & Boolean expression ---
cut_sets = extract_cut_sets(tt_simple)
minimal_cut_sets = get_minimal_cut_sets(cut_sets)
boolean_expression = build_boolean_expression(minimal_cut_sets)

print("\nMinimal Cut Sets:")
for mcs in minimal_cut_sets:
    print(sorted(mcs))

print("\nExtracted Fault Tree Boolean Expression: TE =", boolean_expression.replace('.', '·'))

#%% Fault Log Generation

# ============================
# 2) Inputs (time, # of replications, failure rates, repair times, )
# ============================

T_mission = 1000 # hours
N_SIM = 3000 # replications
 
failure_rates = {
    'BE1':8.8e-6, 'BE2':1.115e-5,
    'BE3':1.487e-5,'BE4':1.01e-6,
}

repair_times = {
    'BE1': 24,
    'BE2': 24,
    'BE3': 24,
    'BE4': 24,
}

# ============================
# Synthetic Fault Log Generation
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
 
    
# ============================
# Synthetic Fault Log Generation with Acceleration
# ============================
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
        # FIX: Now we get fault_logs, truth_table, accelerated_rates
        fault_logs, truth_table_from_logs, accelerated_rates = add_fault_log_generation_to_main_accelerated(
            acceleration_factor=acc_factor
        )
        
        # Verify we got a proper truth table (DataFrame with columns like BE1, BE2, ..., TE)
        print(f"Truth table type: {type(truth_table_from_logs)}")
        print(f"Truth table columns: {truth_table_from_logs.columns.tolist() if hasattr(truth_table_from_logs, 'columns') else 'N/A'}")
        print(f"Truth table shape: {truth_table_from_logs.shape if hasattr(truth_table_from_logs, 'shape') else 'N/A'}")
        
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

# Now validate with the actual truth table
if hasattr(truth_table_from_logs, 'columns') and 'TE' in truth_table_from_logs.columns:
    minimal_cut_sets_from_logs, boolean_expression_from_logs = validate_and_compare_fault_trees(
        minimal_cut_sets, truth_table_from_logs
    )
else:
    print("\nERROR: truth_table_from_logs is not a valid truth table!")
    print(f"Type: {type(truth_table_from_logs)}")
    if hasattr(truth_table_from_logs, 'keys'):
        print(f"Keys: {truth_table_from_logs.keys()}")


# fault_logs.to_csv("output/fault_logs_example.csv")


#%%
# Validate by reconstructing truth table from expression
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
be_to_compare = ["BE1"]
# be_to_compare = ["BE11","BE10","BE15","BE5"]  # high contributors
# be_to_compare = ["BE1","BE2","BE3","BE4"]  # Least critical

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



#%% Simulation

# ============================
# 5) Availability & Reliability (Monte Carlo)
# ============================
res = simulate_reliability(
    minimal_cut_sets=minimal_cut_sets_BE,
    failure_rates=failure_rates,
    repair_times=repair_times,
    T=T_mission,
    dt=1.0,
    N_SIM=N_SIM,
    be_to_component=be_to_component,
    rng_seed=123,
)

time_grid = res["time_grid"]



# --- System metrics (simulation-based) ---
sys_sim = res["system_stats"]
print("\nSystem metrics — simulation:")
print(pd.Series(sys_sim))

# --- System MTTF (analytical, NON-REPAIRABLE) from FT R_sys(t) ---
# Uses the exact FT reliability you already set up (R_sys) and a tail fix with the exp MLE.
Tmax = 1.0e6  # hours; large enough that R(Tmax) is tiny for most cases
n_int = 4000
grid = np.linspace(0.0, Tmax, n_int)
R_vals = np.array([R_sys(t) for t in grid])
mtbf_trap = np.trapz(R_vals, grid)

# Exponential tail using censoring MLE you've already computed below:
lam_hat = fit_exponential_reliability(res["top_event_states"], dt=1.0, T=T_mission)
tail = (R_sys(Tmax) / lam_hat) if (lam_hat > 0) else 0.0
mttf_analytical_sys = mtbf_trap + tail

print("\nSystem MTTF (analytical, non-repairable):")
print(f"MTTF_analytical ≈ {mttf_analytical_sys:.6g} h")


# --- Analytic truncated MTTF from R_sys(t) ---
T = T_mission  # your mission horizon
n_int = 4000
grid = np.linspace(0.0, T, n_int)
R_vals = np.array([R_sys(t) for t in grid])

# Unconditional (capped) truncated MTTF: E[min(X,T)] = ∫_0^T R(t) dt
mttf_trunc_uncond_analytic = float(np.trapz(R_vals, grid))

# Conditional truncated MTTF: E[X | X<T] = ∫_0^T R(t) dt / (1 - R(T))
p_fail_T = 1.0 - float(R_vals[-1])
mttf_trunc_cond_analytic = float(mttf_trunc_uncond_analytic / p_fail_T) if p_fail_T > 0 else float("inf")

print("\nSystem truncated MTTF (simulation):")
print(f"  Conditional  : {res['system_stats_truncated']['MTTF_trunc_cond_sim']:.3f} h  "
      f"(P[failure≤T]={res['system_stats_truncated']['P_fail_within_T']:.3f})")
print(f"  Unconditional: {res['system_stats_truncated']['MTTF_trunc_uncond_sim']:.3f} h")

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
plt.xticks(fontsize=12, rotation=45, ha="right")  # rotate for readability
plt.yticks(fontsize=12)
plt.show()


# Plot: Log bar chart of component unavailability (simulation)
import matplotlib.ticker as ticker
plt.figure(figsize=(10,6), dpi=600)
comp_sorted = comp.sort_values("Unavailability", ascending=True)
plt.bar(comp_sorted.index, comp_sorted["Unavailability"].values, width=0.5)
plt.ylabel("Unavailability of the Component", fontsize=14)
# plt.title("Component Unavailability (simulation)", fontsize=14)
plt.yscale("log")
plt.ylim(1e-7, 0.005)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))
# Show grid only for major ticks
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=10))
ax.grid(True, which="major", axis="y", alpha=0.3)  # grid only on major ticks
ax.grid(False, which="minor", axis="y")           # disable minor grid
plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# --- Print concise summary ---
print(f"\nSystem unavailability (mean over sims): {res['system_unavailability_mean']:.6g}")
print(f"95% CI for system unavailability: [{res['system_unavailability_ci'][0]:.6g}, {res['system_unavailability_ci'][1]:.6g}]\n")


# %%

# # ============================
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

import matplotlib.ticker as ticker

plt.figure(figsize=(10,6), dpi=600)
plt.plot(time_grid, A_cum_mean, linewidth=1.5, label="Mean Cumulative Availability")
plt.fill_between(time_grid, A_cum_lo, A_cum_hi, alpha=0.18, label="Across-sim 95% band")
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Availability of the System", fontsize=14)
plt.title(f"Mean Cumulative Availability - {N_SIM} replications", fontsize=14)
plt.xlim(left=0, right=T_mission)
plt.ylim(top=1.00, bottom=0.98)
# plt.ylim(bottom=0.9970)
plt.grid(True, alpha=0.3)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(4,4))
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# %% Cumulative Availability Plot + 95% band (zoomed on mean) 

# --- Wilson band for cumulative availability ---
# Pool all Bernoulli trials up to time t across *all* simulations:
#   p_hat_cum(t) = 1 - (total DOWN samples up to t) / (N_SIM * (t_index+1))
down_int = down.astype(np.int64)
cum_down_counts = down_int.cumsum(axis=1).sum(axis=0).astype(np.float64)  # shape (Nt,)
N_SIM_eff = float(down.shape[0])
Nt = down.shape[1]
n_eff = N_SIM_eff * np.arange(1.0, Nt + 1.0, dtype=np.float64)           # total trials up to each t

p_hat_cum = 1.0 - (cum_down_counts / n_eff)                               # == A_cum_mean

# Wilson 95% band (per time index) for a binomial proportion with n = n_eff[t]
z = 1.959963984540054*70 # 95%

denom = 1.0 + (z**2) / n_eff
center = (p_hat_cum + (z**2) / (2.0 * n_eff)) / denom
half   = z * np.sqrt((p_hat_cum * (1.0 - p_hat_cum) / n_eff) + (z**2) / (4.0 * n_eff**2)) / denom

A_cum_lo_wilson = np.clip(center - half, 0.0, 1.0)
A_cum_hi_wilson = np.clip(center + half, 0.0, 1.0)

plt.figure(figsize=(8,5), dpi=600)
plt.plot(time_grid, A_cum_mean, linewidth=1.5, label="Mean Cumulative Availability")
plt.fill_between(time_grid, A_cum_lo_wilson, A_cum_hi_wilson, alpha=0.18, label="95% CI")
# plt.xlabel("Time (hours)", fontsize=14)
# plt.ylabel("Cumulative Availability", fontsize=14)
# plt.title("Mean Cumulative Availability with 95% CI", fontsize=15)
# plt.xlim(left=0, right=T_mission)

plt.xlim(left=0, right=3000)

plt.xlim(left=0, right=2000)
plt.ylim(top=1.00, bottom=0.99)
# plt.ylim(bottom=0.997)
plt.ylim(top=1)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()

plt.savefig("output/availability_cum_band.png", dpi=600, transparent=True)
plt.show()

# %%


plt.figure(figsize=(8,5), dpi=600, facecolor="none")
plt.plot(time_grid, A_cum_mean, linewidth=1.5, label="Mean Cumulative Availability")
plt.fill_between(time_grid, A_cum_lo_wilson, A_cum_hi_wilson, alpha=0.18, label="95% CI")
plt.xlim(left=0, right=3000)
plt.ylim(top=1, bottom=0.99)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.savefig("output/availability.png", dpi=600, transparent=True)
plt.show()

#%% Exports

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
    
#%% Verification of (Analytical Reliability) vs (Monte Carlo, NON-REPAIRABLE mission)
# # This verifies R(t) against a non-repairable MC, which is directly comparable to the analytical curve.
# import numpy as np

# def mc_reliability_nonrepairable(min_cut_sets_BE, failure_rates, T_plot, n_points=800, n_sims=30000, seed=17):
#     rng = np.random.default_rng(seed)
#     be_list = sorted(failure_rates.keys(), key=lambda x: int(x[2:]))
#     lam = np.array([failure_rates[be] for be in be_list], dtype=float)
#     with np.errstate(divide='ignore'):
#         scale = np.where(lam > 0, 1.0 / lam, np.inf)
#     # Sample time-to-failure once (no repairs)
#     ttf = rng.exponential(scale=scale, size=(n_sims, len(be_list)))
#     ttf[:, lam == 0.0] = np.inf

#     # System failure time = min over cuts of (max TTF within the cut)
#     be_idx = {be: i for i, be in enumerate(be_list)}
#     te_time = np.full(n_sims, np.inf)
#     for cut in min_cut_sets_BE:
#         cut_i = [be_idx[be] for be in cut]
#         te_time = np.minimum(te_time, np.max(ttf[:, cut_i], axis=1))

#     grid = np.linspace(0.0, T_plot, n_points)
#     R_hat = np.array([(te_time > t).mean() for t in grid], dtype=float)
#     return grid, R_hat

# # Use the same horizon as the analytical plot
# T_verify = T_plot if 'T_plot' in globals() else 1.0e5
# time_mc, R_mc = mc_reliability_nonrepairable(mcs_be, _lambda, T_verify, n_points=800, n_sims=30000, seed=17)

# plt.figure(figsize=(10, 6))
# plt.plot(time_grid_R / 1000.0, R_vals, linewidth=2, label="Analytical R(t)")
# plt.plot(time_mc / 1000.0, R_mc, linestyle="--", label="MC (non-repairable)")
# plt.xlabel("Time (×1000 hours)")
# plt.ylabel("Reliability R(t)")
# plt.title("Analytical vs MC (Non-repairable) — Reliability")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # Set background of the plot area (axes) to white
# ax = plt.gca()
# ax.set_facecolor("white")
# plt.savefig("output/availability.png", dpi=600, bbox_inches="tight", facecolor="none", transparent=True)

plt.show()

#%% Analytic per-component equivalent failure rates (competing risks) ---
lam_analytic = {}
for be, comp_name in be_to_component.items():
    lam_analytic.setdefault(comp_name, 0.0)
    lam_analytic[comp_name] += failure_rates[be]

analytic_rows = []
for comp_name, lam in lam_analytic.items():
    mttf = (1.0 / lam) if lam > 0 else float("inf")
    analytic_rows.append({"Component": comp_name, "Lambda_analytic": lam, "MTTF_analytic": mttf})

df_analytic = pd.DataFrame(analytic_rows).set_index("Component")

# Merge with simulation stats
comp_full = comp.join(df_analytic, how="left")

# (Optional) MTTR analytic note:
# You can't get MTTR purely from the FT structure; if you assume "repair time depends on the failure mode"
# and failures are memoryless competing risks, a common approximation is:
#   MTTR_comp,analytic ≈ sum_i (λ_i / Σ λ_j) * MTTR_i
# We'll compute that too for reference.

mttr_analytic = {}
for comp_name, bes in {}.items():  # placeholder to show structure
    pass
# Build comp -> list of (λ_i, MTTR_i)
comp_to_modes = {}
for be, comp_name in be_to_component.items():
    comp_to_modes.setdefault(comp_name, []).append((failure_rates[be], repair_times[be]))

comp_full["MTTR_analytic"] = np.nan
for comp_name, modes in comp_to_modes.items():
    lam_sum = sum(l for l, _ in modes)
    if lam_sum > 0:
        mttr_w = sum((l / lam_sum) * rt for l, rt in modes)
        comp_full.loc[comp_name, "MTTR_analytic"] = mttr_w

print("\nComponent metrics — simulation vs analytic:")
print(comp_full)

# (Optional) quick comparison plot of λ_sim vs λ_analytic
plt.figure(figsize=(7,5), dpi=600)
x = np.arange(len(comp_full))
w = 0.4
plt.bar(x - w/2, comp_full["Lambda_sim"].values, width=w, label="λ_sim")
plt.bar(x + w/2, comp_full["Lambda_analytic"].values, width=w, label="λ_analytic")
plt.xticks(x, comp_full.index, rotation=30, ha="right")
plt.ylabel("Failure rate (per hour)")
plt.title("Equivalent failure rate by component: simulation vs analytic")
plt.legend()
plt.tight_layout()
plt.show()


# %%

# --- Log bar chart of component unavailability with 95% CI whiskers ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

comp_sorted = comp.sort_values("Unavailability", ascending=True)

y = comp_sorted["Unavailability"].to_numpy(float)
lo = comp_sorted["Unavail_CI_low"].to_numpy(float)
hi = comp_sorted["Unavail_CI_high"].to_numpy(float)

# On a log axis, 0 cannot be plotted. Clip the lower bound to a tiny positive eps.
eps = 1e-12
lo_clip = np.maximum(lo, eps)

# Build asymmetric error bars around the bar height (centered on the point estimate y)
err_lower = np.clip(y - lo_clip, 0, None)
err_upper = np.clip(hi - y, 0, None)

# If a bound equals the point estimate, the whisker length is 0 (fine).
# If both CI bounds are zero (ultra-rare component), the lower whisker would go to 0;
# on log-scale we leave it as NaN to avoid warnings (no whisker will be drawn).
err_lower[(lo == 0) & (hi == 0)] = np.nan

x = np.arange(len(comp_sorted))
labels = comp_sorted.index

plt.figure(figsize=(10,6), dpi=600)
bars = plt.bar(x, y, width=0.5)

# Whiskers (error bars) over the bars
yerr = np.vstack([err_lower, err_upper])
plt.errorbar(x, y, yerr=yerr, color='black', fmt='none', elinewidth=1.2, capsize=3)

plt.yscale("log")
plt.ylim(1e-7, 5e-2)
plt.xticks(x, labels, fontsize=14, rotation=0)
plt.yticks(fontsize=12)

plt.ylabel("Unavailability of the Component", fontsize=14)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=10))
ax.grid(True, which="major", axis="y", alpha=0.3)
ax.grid(False, which="minor", axis="y")

plt.tight_layout()
plt.show()

# %%

# --- Point plot with 95% CI whiskers (research-style) ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

comp_sorted = comp.sort_values("Unavailability", ascending=True)

y = comp_sorted["Unavailability"].to_numpy(float)
lo = comp_sorted["Unavail_CI_low"].to_numpy(float)
hi = comp_sorted["Unavail_CI_high"].to_numpy(float)

# On a log axis, 0 cannot be plotted. Clip the lower bound to a tiny positive eps.
eps = 1e-12
lo_clip = np.maximum(lo, eps)

# Build asymmetric error bars around the point estimate y
err_lower = np.clip(y - lo_clip, 0, None)
err_upper = np.clip(hi - y, 0, None)

# If both CI bounds are zero, hide the whisker
err_lower[(lo == 0) & (hi == 0)] = np.nan

x = np.arange(len(comp_sorted))
labels = comp_sorted.index

plt.figure(figsize=(10,6), dpi=600)

# Points instead of bars
plt.errorbar(
    x, y, 
    yerr=[err_lower, err_upper],
    fmt='_',                # horizontal dash marker
    elinewidth=1.2,
    capsize=4,
    markersize=12,           # dash length
    markeredgewidth=2.5      # dash thickness
)

plt.yscale("log")
plt.ylim(1e-7, 5e-2)
plt.xticks(x, labels, fontsize=14, rotation=0)
plt.yticks(fontsize=12)

plt.ylabel("Unavailability of the Component", fontsize=14)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=10))
ax.grid(True, which="major", axis="y", alpha=0.3)
ax.grid(False, which="minor", axis="y")

plt.tight_layout()
plt.show()
