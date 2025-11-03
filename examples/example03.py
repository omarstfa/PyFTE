# -*- coding: utf-8 -*-
# simplified_fault_tree_analysis.py
# Simple fault tree analysis from fault logs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import math
from typing import List, Dict, Set
import warnings
warnings.filterwarnings('ignore')

class SimpleFaultTreeAnalysis:
    """Simple fault tree analysis from fault logs"""
    
    def __init__(self, fault_logs_path: str):
        self.fault_logs = pd.read_csv(fault_logs_path)
        self.fault_logs['Timestamp'] = pd.to_datetime(self.fault_logs['Timestamp'])
        self.basic_events = sorted(self.fault_logs['Basic_Event'].unique())
        self.minimal_cut_sets = []
        self.failure_rates = {}
        
    def extract_fault_tree(self):
        """Extract minimal cut sets from fault logs"""
        print("Extracting fault tree from fault logs...")
        
        # Get all unique system failure states
        failure_states = set()
        unique_timestamps = sorted(self.fault_logs['Timestamp'].unique())
        
        for timestamp in unique_timestamps:
            # Get events up to this timestamp
            events_up_to_now = self.fault_logs[self.fault_logs['Timestamp'] <= timestamp]
            
            # Determine active basic events
            active_events = set()
            for _, event in events_up_to_now.iterrows():
                if event['Status'] == 'Active':
                    active_events.add(event['Basic_Event'])
                else:
                    active_events.discard(event['Basic_Event'])
            
            # Check if system is failed
            current_events = self.fault_logs[self.fault_logs['Timestamp'] == timestamp]
            if not current_events.empty and current_events['System_Failure'].iloc[-1]:
                failure_states.add(frozenset(active_events))
        
        # Convert to minimal cut sets
        failure_states_list = [set(state) for state in failure_states]
        failure_states_list.sort(key=len)
        
        minimal_cut_sets = []
        for state in failure_states_list:
            is_minimal = True
            for existing_cut in minimal_cut_sets:
                if existing_cut.issubset(state):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_cut_sets.append(state)
        
        self.minimal_cut_sets = minimal_cut_sets
        print(f"Extracted {len(minimal_cut_sets)} minimal cut sets:")
        for cut_set in minimal_cut_sets:
            print(f"  {sorted(cut_set)}")
        
        return minimal_cut_sets
    
    def estimate_parameters(self):
        """Estimate failure rates from fault logs"""
        print("\nEstimating failure rates...")
        
        start_time = self.fault_logs['Timestamp'].min()
        end_time = self.fault_logs['Timestamp'].max()
        total_hours = (end_time - start_time).total_seconds() / 3600
        
        for be in self.basic_events:
            be_events = self.fault_logs[self.fault_logs['Basic_Event'] == be]
            failures = be_events[be_events['Status'] == 'Active']
            num_failures = len(failures)
            
            failure_rate = num_failures / total_hours if total_hours > 0 else 0
            self.failure_rates[be] = failure_rate
            
            print(f"  {be}: {num_failures} failures, λ = {failure_rate:.2e}/hr")
        
        return self.failure_rates
    
    def analytical_reliability(self, t_hours: float) -> float:
        """Calculate system reliability analytically"""
        if not self.minimal_cut_sets:
            self.extract_fault_tree()
        
        R_system = 1.0
        
        for cut_set in self.minimal_cut_sets:
            cut_failure_prob = 1.0
            for be in cut_set:
                lambda_be = self.failure_rates[be]
                F_be = 1 - math.exp(-lambda_be * t_hours)
                cut_failure_prob *= F_be
            
            R_system *= (1 - cut_failure_prob)
        
        return R_system
    
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
    
    def monte_carlo_simulation(self, T_mission: float = 1000, N_sim: int = 1000, dt: float = 10.0):
        """Run Monte Carlo simulation for repairable system - for availability calculation"""
        np.random.seed(42)
        
        time_grid = np.arange(0, T_mission + dt, dt)
        N_time = len(time_grid)
        
        # Storage for results
        system_states = np.zeros((N_sim, N_time), dtype=bool)
        component_states = {be: np.zeros((N_sim, N_time), dtype=bool) for be in self.basic_events}
        
        # Use estimated repair times (default 24 hours)
        repair_times = {be: 24 for be in self.basic_events}
        
        for sim in range(N_sim):
            # Simulate each component
            for be, lambda_rate in self.failure_rates.items():
                repair_time = repair_times[be]
                component_states[be][sim, :] = self._simulate_component(
                    lambda_rate, repair_time, T_mission, dt)
            
            # Determine system state at each time point
            for t_idx in range(N_time):
                system_failed = False
                
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
        
        # Calculate availability
        availability = 1 - system_states.mean(axis=0)
        
        # Component metrics
        component_metrics = {}
        for be in self.basic_events:
            comp_unavailability = component_states[be].mean()
            component_metrics[be] = {
                'unavailability': comp_unavailability,
                'lambda': self.failure_rates[be]
            }
        
        results = {
            'time_grid': time_grid,
            'availability': availability,
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
                ttf = np.random.exponential(1/lambda_rate) if lambda_rate > 0 else float('inf')
                next_event_time = current_time + ttf
                next_event_type = 'failure'
            else:
                next_event_time = current_time + repair_time
                next_event_type = 'repair'
            
            event_idx = int(current_time / dt)
            if next_event_time >= T_mission:
                end_idx = time_points
            else:
                end_idx = int(next_event_time / dt)
            
            state[event_idx:end_idx] = current_state
            
            if next_event_time >= T_mission:
                break
                
            current_time = next_event_time
            current_state = (next_event_type == 'failure')
        
        return state
    
    def plot_simple_analysis(self, mission_time: float = 1000):
        """Create the 4 required plots in one figure"""
        # Run simulation for availability data
        sim_results = self.monte_carlo_simulation(mission_time, N_sim=500, dt=5.0)
        
        time_grid = sim_results['time_grid']
        
        # System reliability over time (analytical)
        system_reliability = [self.analytical_reliability(t) for t in time_grid]
        
        # Basic event reliabilities
        basic_event_reliability = {}
        for be, lambda_rate in self.failure_rates.items():
            basic_event_reliability[be] = [math.exp(-lambda_rate * t) for t in time_grid]
        
        # Structural importance
        structural_importance = self.calculate_structural_importance()
        
        # Create the 4-plot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Reliability vs Availability (from example01.py style)
        ax1.plot(time_grid, system_reliability, 'b-', linewidth=2, label='Reliability R(t)')
        ax1.plot(time_grid, sim_results['availability'], 'r-', linewidth=2, label='Availability A(t)')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Probability')
        ax1.set_title('System Reliability vs Availability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: System vs Basic Event Reliability
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
        
        # Plot 3: Structural Importance
        components = list(structural_importance.keys())
        importance_values = [structural_importance[be] for be in components]
        bars = ax3.bar(components, importance_values, color='lightblue', alpha=0.7)
        ax3.set_xlabel('Basic Event')
        ax3.set_ylabel('Structural Importance')
        ax3.set_title('Structural Importance of Basic Events')
        
        # Add value labels on bars
        for bar, value in zip(bars, importance_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Component Unavailability (from example01.py style)
        components_avail = list(sim_results['component_metrics'].keys())
        unavail_values = [sim_results['component_metrics'][be]['unavailability'] for be in components_avail]
        bars = ax4.bar(components_avail, unavail_values, color='lightcoral', alpha=0.7)
        ax4.set_xlabel('Component')
        ax4.set_ylabel('Unavailability')
        ax4.set_title('Component Unavailability')
        
        # Add value labels on bars
        for bar, value in zip(bars, unavail_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def run_simple_analysis(fault_logs_path: str):
    """Run the simplified analysis"""
    print("=" * 60)
    print("SIMPLE FAULT TREE ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SimpleFaultTreeAnalysis(fault_logs_path)
    
    # Step 1: Extract fault tree
    print("\n1. FAULT TREE EXTRACTION")
    print("-" * 30)
    minimal_cut_sets = analyzer.extract_fault_tree()
    
    # Step 2: Estimate parameters
    print("\n2. PARAMETER ESTIMATION")
    print("-" * 30)
    failure_rates = analyzer.estimate_parameters()
    
    # Step 3: Calculate structural importance
    print("\n3. STRUCTURAL IMPORTANCE")
    print("-" * 30)
    structural_importance = analyzer.calculate_structural_importance()
    for be, importance in structural_importance.items():
        print(f"  {be}: {importance:.3f}")
    
    # Step 4: Reliability calculations
    print("\n4. RELIABILITY ANALYSIS")
    print("-" * 30)
    mission_time = 1000
    final_reliability = analyzer.analytical_reliability(mission_time)
    print(f"System reliability at {mission_time} hours: {final_reliability:.6f}")
    
    print("\nBasic event reliabilities:")
    for be, lambda_rate in failure_rates.items():
        R_be = math.exp(-lambda_rate * mission_time)
        print(f"  {be}: {R_be:.6f}")
    
    # Step 5: Create plots
    print("\n5. GENERATING PLOTS")
    print("-" * 30)
    analyzer.plot_simple_analysis(mission_time)
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Minimal Cut Sets: {len(minimal_cut_sets)}")
    print(f"Basic Events: {analyzer.basic_events}")
    print(f"Most Structurally Important: {max(structural_importance, key=structural_importance.get)}")
    print(f"Final System Reliability: {final_reliability:.4f}")
    
    return analyzer

# Run the analysis
if __name__ == "__main__":
    fault_logs_path = "output/fault_logs_example.csv"
    
    try:
        analyzer = run_simple_analysis(fault_logs_path)
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Additional helper function to show the boolean expression
def show_boolean_expression(analyzer):
    """Display the extracted boolean expression"""
    if analyzer.minimal_cut_sets:
        print("\nEXTRACTED BOOLEAN EXPRESSION:")
        print("-" * 30)
        terms = []
        for cut_set in analyzer.minimal_cut_sets:
            if len(cut_set) == 1:
                terms.append(list(cut_set)[0])
            else:
                terms.append(f"({' · '.join(sorted(cut_set))})")
        boolean_expr = " + ".join(terms)
        print(f"Top Event = {boolean_expr}")

# Show boolean expression if analyzer exists
if 'analyzer' in locals():
    show_boolean_expression(analyzer)