# simple_fault_tree_analysis.py
# Simplified fault tree analysis from fault logs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import math

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
        print("Step 1: Extracting fault tree structure from logs...")
        
        # Get system failure states
        failure_states = set()
        unique_timestamps = sorted(self.fault_logs['Timestamp'].unique())
        
        for timestamp in unique_timestamps:
            # Get active events at this timestamp
            events_up_to_now = self.fault_logs[self.fault_logs['Timestamp'] <= timestamp]
            active_events = set()
            
            for _, event in events_up_to_now.iterrows():
                if event['Status'] == 'Active':
                    active_events.add(event['Basic_Event'])
                else:
                    active_events.discard(event['Basic_Event'])
            
            # Check if system failed
            current_events = self.fault_logs[self.fault_logs['Timestamp'] == timestamp]
            if not current_events.empty and current_events['System_Failure'].iloc[-1]:
                failure_states.add(frozenset(active_events))
        
        # Find minimal cut sets
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
        
        print("✓ Minimal Cut Sets Found:")
        for i, cut_set in enumerate(minimal_cut_sets, 1):
            print(f"  Cut Set {i}: {sorted(cut_set)}")
        
        return minimal_cut_sets
    
    def estimate_failure_rates(self):
        """Estimate failure rates from fault logs"""
        print("\nStep 2: Estimating failure rates from logs...")
        
        start_time = self.fault_logs['Timestamp'].min()
        end_time = self.fault_logs['Timestamp'].max()
        total_hours = (end_time - start_time).total_seconds() / 3600
        
        for be in self.basic_events:
            be_events = self.fault_logs[self.fault_logs['Basic_Event'] == be]
            failures = be_events[be_events['Status'] == 'Active']
            num_failures = len(failures)
            
            failure_rate = num_failures / total_hours if total_hours > 0 else 0
            self.failure_rates[be] = failure_rate
            
            print(f"  {be}: {num_failures} failures → λ = {failure_rate:.2e}/hour")
        
        return self.failure_rates
    
    def calculate_structural_importance(self):
        """Calculate structural importance of basic events"""
        print("\nStep 3: Calculating structural importance...")
        
        importance = {}
        total_cut_sets = len(self.minimal_cut_sets)
        
        for be in self.basic_events:
            # Count how many cut sets contain this basic event
            cut_sets_with_be = sum(1 for cut_set in self.minimal_cut_sets if be in cut_set)
            structural_importance = cut_sets_with_be / total_cut_sets if total_cut_sets > 0 else 0
            
            importance[be] = structural_importance
            
            print(f"  {be}: appears in {cut_sets_with_be}/{total_cut_sets} cut sets → importance = {structural_importance:.3f}")
        
        return importance
    
    def analytical_reliability(self, t_hours: float) -> float:
        """Calculate system reliability (non-repairable system)"""
        R_system = 1.0
        
        for cut_set in self.minimal_cut_sets:
            # Probability that this cut set fails
            cut_failure_prob = 1.0
            for be in cut_set:
                lambda_be = self.failure_rates[be]
                F_be = 1 - math.exp(-lambda_be * t_hours)
                cut_failure_prob *= F_be
            
            # System survives if no cut set fails
            R_system *= (1 - cut_failure_prob)
        
        return R_system
    
    def basic_event_reliability(self, be: str, t_hours: float) -> float:
        """Calculate reliability of a single basic event"""
        lambda_be = self.failure_rates[be]
        return math.exp(-lambda_be * t_hours)
    
    def plot_reliability_curves(self, max_time: float = 1000):
        """Plot reliability curves for system and basic events"""
        print(f"\nStep 4: Plotting reliability curves up to {max_time} hours...")
        
        time_points = np.linspace(0, max_time, 100)
        
        # Calculate reliabilities
        system_reliability = [self.analytical_reliability(t) for t in time_points]
        be_reliabilities = {}
        
        for be in self.basic_events:
            be_reliabilities[be] = [self.basic_event_reliability(be, t) for t in time_points]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot system reliability
        plt.plot(time_points, system_reliability, 'k-', linewidth=3, label='System (Top Event)')
        
        # Plot basic event reliabilities
        colors = ['red', 'blue', 'green', 'orange']
        for i, be in enumerate(self.basic_events):
            plt.plot(time_points, be_reliabilities[be], '--', 
                    color=colors[i % len(colors)], linewidth=2, label=f'{be}')
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Reliability')
        plt.title('System vs Basic Event Reliability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("✓ Reliability plot generated")
    
    def print_reliability_table(self):
        """Print reliability values at key time points"""
        print("\nStep 5: Reliability at key time points")
        print("-" * 50)
        print(f"{'Time (hours)':<12} {'System':<10} {''.join([f'{be:<10}' for be in self.basic_events])}")
        print("-" * 50)
        
        time_points = [0, 100, 250, 500, 750, 1000]
        for t in time_points:
            system_r = self.analytical_reliability(t)
            be_rs = [self.basic_event_reliability(be, t) for be in self.basic_events]
            
            print(f"{t:<12} {system_r:<10.4f} {''.join([f'{r:<10.4f}' for r in be_rs])}")

def run_simple_analysis(fault_logs_path: str):
    """Run the complete simplified analysis"""
    print("=" * 60)
    print("SIMPLE FAULT TREE ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SimpleFaultTreeAnalysis(fault_logs_path)
    
    # Step 1: Extract fault tree
    analyzer.extract_fault_tree()
    
    # Step 2: Estimate parameters
    analyzer.estimate_failure_rates()
    
    # Step 3: Structural importance
    importance = analyzer.calculate_structural_importance()
    
    # Step 4: Plot reliability curves
    analyzer.plot_reliability_curves(max_time=1000)
    
    # Step 5: Print reliability table
    analyzer.print_reliability_table()
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    most_important = max(importance.items(), key=lambda x: x[1])
    least_important = min(importance.items(), key=lambda x: x[1])
    
    print(f"Most important component: {most_important[0]} (importance: {most_important[1]:.3f})")
    print(f"Least important component: {least_important[0]} (importance: {least_important[1]:.3f})")
    
    # Boolean expression
    if analyzer.minimal_cut_sets:
        print(f"\nSystem failure occurs when:")
        for i, cut_set in enumerate(analyzer.minimal_cut_sets, 1):
            if len(cut_set) == 1:
                print(f"  {i}. {list(cut_set)[0]} fails")
            else:
                components = " AND ".join(sorted(cut_set))
                print(f"  {i}. {components} fail together")
    
    return analyzer

# Run the analysis
if __name__ == "__main__":
    fault_logs_path = "output/fault_logs_example.csv"
    
    try:
        analyzer = run_simple_analysis(fault_logs_path)
        print("\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")