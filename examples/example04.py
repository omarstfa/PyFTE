# -*- coding: utf-8 -*-
# analyze_from_fault_logs.py
# Extract fault tree from fault logs and perform complete analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import math
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class FaultTreeFromLogs:
    """Extract fault tree from fault logs and perform reliability analysis"""
    
    def __init__(self, fault_logs_path: str):
        self.fault_logs = pd.read_csv(fault_logs_path)
        self.fault_logs['Timestamp'] = pd.to_datetime(self.fault_logs['Timestamp'])
        self.basic_events = sorted(self.fault_logs['Basic_Event'].unique())
        self.minimal_cut_sets = []
        self.failure_rates = {}
        self.repair_times = {}
        
    def extract_fault_tree(self):
        """Extract minimal cut sets from fault logs"""
        print("Extracting fault tree from fault logs...")
        
        # Get all unique system failure states
        failure_states = set()
        
        # Process each timestamp to get system state
        unique_timestamps = sorted(self.fault_logs['Timestamp'].unique())
        
        for timestamp in unique_timestamps:
            # Get events up to this timestamp
            events_up_to_now = self.fault_logs[self.fault_logs['Timestamp'] <= timestamp]
            
            # Determine active basic events at this timestamp
            active_events = set()
            for _, event in events_up_to_now.iterrows():
                if event['Status'] == 'Active':
                    active_events.add(event['Basic_Event'])
                else:  # 'Cleared'
                    active_events.discard(event['Basic_Event'])
            
            # Check if system is failed at this timestamp
            current_events = self.fault_logs[self.fault_logs['Timestamp'] == timestamp]
            if not current_events.empty and current_events['System_Failure'].iloc[-1]:
                failure_states.add(frozenset(active_events))
        
        print(f"Found {len(failure_states)} unique failure states")
        
        # Convert to minimal cut sets
        failure_states_list = [set(state) for state in failure_states]
        failure_states_list.sort(key=len)  # Sort by size
        
        minimal_cut_sets = []
        for state in failure_states_list:
            # Check if this state is minimal (no subset is already a cut set)
            is_minimal = True
            for existing_cut in minimal_cut_sets:
                if existing_cut.issubset(state):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_cut_sets.append(state)
        
        self.minimal_cut_sets = minimal_cut_sets
        print(f"Extracted {len(minimal_cut_sets)} minimal cut sets:")
        for i, cut_set in enumerate(minimal_cut_sets, 1):
            print(f"  Cut Set {i}: {sorted(cut_set)}")
        
        return minimal_cut_sets
    
    def estimate_parameters(self):
        """Estimate failure rates and repair times from fault logs"""
        print("\nEstimating parameters from fault logs...")
        
        # Calculate total observation time
        start_time = self.fault_logs['Timestamp'].min()
        end_time = self.fault_logs['Timestamp'].max()
        total_hours = (end_time - start_time).total_seconds() / 3600
        print(f"Total observation period: {total_hours:.2f} hours")
        
        for be in self.basic_events:
            be_events = self.fault_logs[self.fault_logs['Basic_Event'] == be]
            
            # Count failures (Active events)
            failures = be_events[be_events['Status'] == 'Active']
            num_failures = len(failures)
            
            # Estimate failure rate (failures per hour)
            failure_rate = num_failures / total_hours if total_hours > 0 else 0
            
            # Estimate repair time from Active->Cleared transitions
            repair_times = []
            for i in range(len(be_events)-1):
                if (be_events.iloc[i]['Status'] == 'Active' and 
                    be_events.iloc[i+1]['Status'] == 'Cleared' and
                    be_events.iloc[i+1]['Basic_Event'] == be):
                    repair_duration = (be_events.iloc[i+1]['Timestamp'] - be_events.iloc[i]['Timestamp']).total_seconds() / 3600
                    repair_times.append(repair_duration)
            
            avg_repair_time = np.mean(repair_times) if repair_times else 24  # Default 24 hours
            
            self.failure_rates[be] = failure_rate
            self.repair_times[be] = avg_repair_time
            
            print(f"  {be}: {num_failures} failures, λ = {failure_rate:.2e}/hr, MTTR = {avg_repair_time:.1f} hr")
        
        return self.failure_rates, self.repair_times
    
    def analytical_reliability(self, t_hours: float) -> float:
        """Calculate system reliability analytically (non-repairable)"""
        if not self.minimal_cut_sets:
            self.extract_fault_tree()
        
        R_system = 1.0
        
        for cut_set in self.minimal_cut_sets:
            # Probability that this cut set fails
            cut_failure_prob = 1.0
            for be in cut_set:
                lambda_be = self.failure_rates[be]
                F_be = 1 - math.exp(-lambda_be * t_hours)  # Failure probability
                cut_failure_prob *= F_be
            
            # System survives if no cut set fails
            R_system *= (1 - cut_failure_prob)
        
        return R_system
    
    def monte_carlo_simulation(self, T_mission: float = 1000, N_sim: int = 1000, dt: float = 10.0):
        """Run Monte Carlo simulation for repairable system"""
        np.random.seed(42)
        
        time_grid = np.arange(0, T_mission + dt, dt)
        N_time = len(time_grid)
        
        # Storage for results
        system_states = np.zeros((N_sim, N_time), dtype=bool)
        component_states = {be: np.zeros((N_sim, N_time), dtype=bool) for be in self.basic_events}
        
        print(f"\nRunning {N_sim} Monte Carlo simulations...")
        
        for sim in range(N_sim):
            # Simulate each component
            for be, lambda_rate in self.failure_rates.items():
                repair_time = self.repair_times[be]
                component_states[be][sim, :] = self._simulate_component(
                    lambda_rate, repair_time, T_mission, dt)
            
            # Determine system state at each time point
            for t_idx in range(N_time):
                system_failed = False
                
                # Check each minimal cut set
                for cut_set in self.minimal_cut_sets:
                    cut_failed = True
                    for be in cut_set:
                        if not component_states[be][sim, t_idx]:
                            cut_failed = False
                            break
                    
                    if cut_failed:
                        system_failed = True
                        break
                
                system_states[sim, t_idx] = system_failed
        
        # Calculate results
        reliability = 1 - np.logical_or.accumulate(system_states, axis=1).mean(axis=0)
        availability = 1 - system_states.mean(axis=0)
        
        # System metrics
        system_unavailability = system_states.mean()
        total_failures = np.sum(np.diff(system_states.astype(int), axis=1) > 0)
        
        # Component metrics
        component_metrics = {}
        for be in self.basic_events:
            comp_unavailability = component_states[be].mean()
            component_metrics[be] = {
                'unavailability': comp_unavailability,
                'lambda': self.failure_rates[be],
                'repair_time': self.repair_times[be]
            }
        
        results = {
            'time_grid': time_grid,
            'reliability': reliability,
            'availability': availability,
            'system_states': system_states,
            'component_states': component_states,
            'system_unavailability': system_unavailability,
            'total_failures': total_failures,
            'component_metrics': component_metrics
        }
        
        return results
    
    def _simulate_component(self, lambda_rate: float, repair_time: float, 
                          T_mission: float, dt: float) -> np.ndarray:
        """Simulate a single component's failure/repair timeline"""
        time_points = int(T_mission / dt) + 1
        state = np.zeros(time_points, dtype=bool)
        
        current_time = 0.0
        current_state = False
        
        while current_time < T_mission:
            if not current_state:
                # Component is operational - time to next failure
                ttf = np.random.exponential(1/lambda_rate) if lambda_rate > 0 else float('inf')
                next_event_time = current_time + ttf
                next_event_type = 'failure'
            else:
                # Component is failed - time to repair completion
                next_event_time = current_time + repair_time
                next_event_type = 'repair'
            
            # Determine state for time segments
            event_idx = int(current_time / dt)
            if next_event_time >= T_mission:
                end_idx = time_points
            else:
                end_idx = int(next_event_time / dt)
            
            state[event_idx:end_idx] = current_state
            
            if next_event_time >= T_mission:
                break
                
            # Update for next event
            current_time = next_event_time
            current_state = (next_event_type == 'failure')
        
        return state
    
    def calculate_importance_factors(self, mission_time: float = 1000):
        """Calculate Birnbaum importance factors"""
        if not self.minimal_cut_sets:
            self.extract_fault_tree()
        
        # Baseline system unreliability
        Q_base = 1 - self.analytical_reliability(mission_time)
        
        importance = {}
        
        for be in self.basic_events:
            # Calculate system unreliability with this BE perfectly reliable
            original_rate = self.failure_rates[be]
            self.failure_rates[be] = 0  # Perfect component
            Q_without_be = 1 - self.analytical_reliability(mission_time)
            
            # Restore original rate
            self.failure_rates[be] = original_rate
            
            # Birnbaum importance
            birnbaum = (Q_base - Q_without_be) 
            
            importance[be] = {
                'birnbaum': birnbaum,
                'failure_rate': original_rate,
                'contribution': birnbaum * original_rate
            }
        
        return importance
    
    def calculate_structural_importance(self):
        """Calculate structural importance of each basic event"""
        if not self.minimal_cut_sets:
            self.extract_fault_tree()
        
        def prod(values):
            result = 1.0
            for v in values:
                result *= v
            return result
        
        structure_importance = {}
        
        for be in self.basic_events:
            terms = []
            for cut_set in self.minimal_cut_sets:
                if be in cut_set:
                    Nj = len(cut_set)
                    terms.append(1 - 1 / (2 ** (Nj - 1)))
            
            I_phi = 1 - prod(terms) if terms else 0
            structure_importance[be] = I_phi
        
        return structure_importance
    
    def plot_analysis(self, results: dict):
        """Create comprehensive analysis plots"""
        time_grid = results['time_grid']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Reliability vs Availability
        ax1.plot(time_grid, results['reliability'], 'b-', linewidth=2, label='Reliability R(t)')
        ax1.plot(time_grid, results['availability'], 'r-', linewidth=2, label='Availability A(t)')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Probability')
        ax1.set_title('System Reliability vs Availability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: System vs Basic Event Reliability (from example03)
        system_reliability = [self.analytical_reliability(t) for t in time_grid]
        
        # Basic event reliabilities
        basic_event_reliability = {}
        for be, lambda_rate in self.failure_rates.items():
            basic_event_reliability[be] = [math.exp(-lambda_rate * t) for t in time_grid]
        
        ax2.plot(time_grid, system_reliability, 'k-', linewidth=3, label='System (Top Event)')
        colors = ['red', 'blue', 'green', 'orange']
        for i, (be, rel_values) in enumerate(basic_event_reliability.items()):
            ax2.plot(time_grid, rel_values, '--', linewidth=2, 
                    color=colors[i % len(colors)], label=be)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Reliability')
        ax2.set_title('System vs Basic Event Reliability')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Component Unavailability
        components = list(results['component_metrics'].keys())
        unavailabilities = [results['component_metrics'][be]['unavailability'] for be in components]
        bars = ax3.bar(components, unavailabilities)
        ax3.set_xlabel('Component')
        ax3.set_ylabel('Unavailability')
        ax3.set_title('Component Unavailability')
        
        # Add value labels on bars
        for bar, value in zip(bars, unavailabilities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Structural Importance of Basic Events (from example03)
        structural_importance = self.calculate_structural_importance()
        components = list(structural_importance.keys())
        importance_values = [structural_importance[be] for be in components]
        bars = ax4.bar(components, importance_values, color='lightblue', alpha=0.7)
        ax4.set_xlabel('Basic Event')
        ax4.set_ylabel('Structural Importance')
        ax4.set_title('Structural Importance of Basic Events')
        
        # Add value labels on bars
        for bar, value in zip(bars, importance_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def run_complete_analysis(fault_logs_path: str):
    """Run complete analysis from fault logs"""
    print("=" * 70)
    print("COMPLETE FAULT TREE ANALYSIS FROM FAULT LOGS")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = FaultTreeFromLogs(fault_logs_path)
    
    # Step 1: Extract fault tree structure
    print("\n1. EXTRACTING FAULT TREE STRUCTURE")
    print("-" * 40)
    minimal_cut_sets = analyzer.extract_fault_tree()
    
    # Step 2: Estimate parameters
    print("\n2. ESTIMATING PARAMETERS")
    print("-" * 40)
    failure_rates, repair_times = analyzer.estimate_parameters()
    
    # Step 3: Analytical reliability
    print("\n3. ANALYTICAL RELIABILITY ANALYSIS")
    print("-" * 40)
    time_points = [0, 100, 250, 500, 750, 1000]
    print("Reliability at different times (non-repairable system):")
    for t in time_points:
        R_t = analyzer.analytical_reliability(t)
        print(f"  R({t:4d} hours) = {R_t:.6f}")
    
    # Step 4: Monte Carlo simulation
    print("\n4. MONTE CARLO SIMULATION")
    print("-" * 40)
    results = analyzer.monte_carlo_simulation(T_mission=1000, N_sim=1000, dt=5.0)
    
    # Print key results
    print("\nSimulation Results:")
    print(f"Final Reliability: {results['reliability'][-1]:.6f}")
    print(f"Final Availability: {results['availability'][-1]:.6f}")
    print(f"System Unavailability: {results['system_unavailability']:.6f}")
    print(f"Total System Failures: {results['total_failures']}")
    
    print("\nComponent Metrics:")
    for be, metrics in results['component_metrics'].items():
        print(f"  {be}: Unavailability = {metrics['unavailability']:.2e}, "
              f"λ = {metrics['lambda']:.2e}/hr, MTTR = {metrics['repair_time']:.1f} hr")
    
    # Step 5: Importance factors
    print("\n5. IMPORTANCE FACTOR ANALYSIS")
    print("-" * 40)
    importance = analyzer.calculate_importance_factors()
    print("Birnbaum Importance Factors:")
    for be, factors in importance.items():
        print(f"  {be}: {factors['birnbaum']:.4f} "
              f"(failure rate: {factors['failure_rate']:.2e}/hr)")
    
    # Step 6: Structural importance
    print("\n6. STRUCTURAL IMPORTANCE ANALYSIS")
    print("-" * 40)
    structural_importance = analyzer.calculate_structural_importance()
    for be, importance_val in structural_importance.items():
        print(f"  {be}: {importance_val:.3f}")
    
    # Step 7: Create plots
    print("\n7. GENERATING ANALYSIS PLOTS")
    print("-" * 40)
    analyzer.plot_analysis(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Extracted Minimal Cut Sets: {len(minimal_cut_sets)}")
    print(f"Basic Events: {analyzer.basic_events}")
    print("Mission Time: 1,000 hours")
    print(f"Final System Availability: {results['availability'][-1]:.4f}")
    
    most_critical = max(importance.items(), key=lambda x: x[1]['birnbaum'])[0]
    print(f"Most Critical Component (Birnbaum): {most_critical}")
    
    most_structural = max(structural_importance.items(), key=lambda x: x[1])[0]
    print(f"Most Structurally Important: {most_structural}")
    
    # Boolean expression
    if minimal_cut_sets:
        print("\nBoolean Expression (approximate):")
        terms = []
        for cut_set in minimal_cut_sets:
            if len(cut_set) == 1:
                terms.append(list(cut_set)[0])
            else:
                terms.append(f"({' · '.join(sorted(cut_set))})")
        boolean_expr = " + ".join(terms)
        print(f"TE = {boolean_expr}")
    
    return analyzer, results

# Run the analysis
if __name__ == "__main__":
    fault_logs_path = "fault_logs_example.csv"
    
    try:
        analyzer, results = run_complete_analysis(fault_logs_path)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

# Additional detailed analysis
def detailed_cut_set_analysis(analyzer):
    """Perform detailed cut set analysis"""
    print("\n" + "=" * 70)
    print("DETAILED CUT SET ANALYSIS")
    print("=" * 70)
    
    cut_sets = analyzer.minimal_cut_sets
    failure_rates = analyzer.failure_rates
    
    print("Cut Set Contributions:")
    for i, cut_set in enumerate(cut_sets, 1):
        # Probability of cut set failure over mission time
        cut_prob = 1.0
        for be in cut_set:
            lambda_be = failure_rates[be]
            F_be = 1 - math.exp(-lambda_be * 1000)  # 1000-hour mission
            cut_prob *= F_be
        
        print(f"Cut Set {i}: {sorted(cut_set)}")
        print(f"  Probability of failure: {cut_prob:.6f}")
        print(f"  Components: {', '.join([f'{be}(λ={failure_rates[be]:.2e})' for be in cut_set])}")
        
        # Importance of this cut set
        if len(cut_set) > 1:
            print("  Type: AND gate (requires all components to fail)")
        else:
            print("  Type: Single point failure")

# Run detailed analysis
if 'analyzer' in locals():
    detailed_cut_set_analysis(analyzer)
    
#%% Seperate Plots
# =============================================================================
# Imports (once)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# =============================================================================
# Plot 1 : Reliability vs Availability
# =============================================================================
t = results["time_grid"]
rel = results["reliability"]
avail = results["availability"]

fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
ax.plot(t, rel, linewidth=2, label="Reliability (no failure yet)")
ax.plot(t, avail, linewidth=2, label="Availability (up at time t)")
ax.set_xlabel("Time (hours)", fontsize=14)
ax.set_ylabel("Probability", fontsize=14)
ax.set_title("System Reliability vs Availability", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()

# =============================================================================
# Plot 2 : System vs Basic-Event Reliability (no repairs)
# =============================================================================
t = results["time_grid"]
system_rel = np.array([analyzer.analytical_reliability(float(x)) for x in t])

fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
ax.plot(t, system_rel, linewidth=3, label="System (top event)", color="black")
for be, lam in analyzer.failure_rates.items():
    ax.plot(t, np.exp(-lam * t), "--", linewidth=2, label=be)
ax.set_xlabel("Time (hours)", fontsize=14)
ax.set_ylabel("Reliability", fontsize=14)
ax.set_title("System vs Basic-Event Reliability (no repairs)", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
# ax.set_xlim(0, 200)
fig.tight_layout()
plt.show()

# =============================================================================
# Plot 3 : Component Unavailability (simulation-based)
# =============================================================================
metrics = results["component_metrics"]
components = list(metrics.keys())
unavail = [metrics[be]["unavailability"] for be in components]

fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
bars = ax.bar(components, unavail)
ax.set_xlabel("Component", fontsize=14)
ax.set_ylabel("Unavailability (fraction of time down)", fontsize=14)
ax.set_title("Component Unavailability (simulation-based)", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(0, 0.35)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.ticklabel_format(axis='y', style='plain', useOffset=False)

for b, v in zip(bars, unavail):
    if v >= 0.01:
        label = f"{v*100:.2f}%"
    elif v >= 0.0001:
        label = f"{v*100:.4f}%"
    else:
        label = f"{v*100:.6f}%"
    ax.text(b.get_x() + b.get_width()/2, b.get_height(), label,
            ha="center", va="bottom", fontsize=10)

fig.tight_layout()
plt.show()

# =============================================================================
# Plot 4 : Structural Importance (logic-only)
# =============================================================================
struct_import = analyzer.calculate_structural_importance()
components = list(struct_import.keys())
scores = [struct_import[be] for be in components]

fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
bars = ax.bar(components, scores)
ax.set_xlabel("Basic event", fontsize=14)
ax.set_ylabel("Structural importance", fontsize=14)
ax.set_title("Structural Importance of Basic Events", fontsize=14)
ax.tick_params(axis='both', labelsize=12)

for b, v in zip(bars, scores):
    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
            f"{v:.3f}", ha="center", va="bottom", fontsize=10)

fig.tight_layout()
plt.show()

