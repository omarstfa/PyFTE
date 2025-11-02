# fault_log_generator.py - FIXED VERSION

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
# fault_log_generator.py - FIXED VERSION

def generate_synthetic_fault_logs_accelerated(failure_rates, repair_times, minimal_cut_sets, 
                                            mission_time_hours=8760, acceleration_factor=1000, 
                                            seed=42):
    """
    Generate synthetic fault logs with proper state tracking
    """
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Apply acceleration factor to failure rates
    accelerated_rates = {be: rate * acceleration_factor for be, rate in failure_rates.items()}
    
    print(f"Using acceleration factor: {acceleration_factor}")
    print(f"Repair times: {repair_times}")
    
    # BE to description mapping
    be_descriptions = {
        'BE1': "Fuse aging fault detected",
        'BE2': "Fuse installation fault detected", 
        'BE3': "MCB 1 fault detected",
        'BE4': "MCB 2 fault detected",
        'BE5': "PV interconnect fault detected",
        'BE6': "Grounding system fault detected",
        'BE7': "PV glass breakage detected",
        'BE8': "PV soiling detected",
        'BE9': "PV shading detected",
        'BE10': "PV cell breakage detected",
        'BE11': "PV solder bond failure detected",
        'BE12': "PV hot spot detected",
        'BE13': "Diode fault detected",
        'BE14': "Short/open circuit fault detected",
        'BE15': "Rack structure fault detected",
        'BE16': "Encapsulant fault detected",
        'BE17': "Cable insulation fault detected",
        'BE18': "Cable aging fault detected"
    }
    
    # Start time
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    
    # Track state for each BE: {'BE1': 'operational', 'BE2': 'failed', ...}
    be_states = {be: 'operational' for be in failure_rates.keys()}
    
    # Track next available time for each BE (when it becomes operational again)
    next_available_time = {be: 0.0 for be in failure_rates.keys()}
    
    all_events = []
    
    # Generate events using event-based simulation
    current_time = 0.0
    
    while current_time < mission_time_hours:
        # Find the next event (failure or repair completion)
        next_events = []
        
        for be, state in be_states.items():
            if state == 'operational':
                # Generate time to next failure
                lambda_rate = accelerated_rates[be]
                if lambda_rate > 0:
                    ttf = np.random.exponential(1/lambda_rate)
                    next_events.append(('failure', be, current_time + ttf))
            else:  # state == 'failed'
                # Repair completion is already scheduled in next_available_time
                next_events.append(('repair_complete', be, next_available_time[be]))
        
        if not next_events:
            break
            
        # Find the next event to process
        next_events.sort(key=lambda x: x[2])  # Sort by time
        next_event_type, next_be, next_time = next_events[0]
        
        if next_time >= mission_time_hours:
            break
            
        current_time = next_time
        
        if next_event_type == 'failure':
            # BE fails
            if be_states[next_be] == 'operational':  # Double-check it's still operational
                be_states[next_be] = 'failed'
                repair_duration = repair_times[next_be]
                next_available_time[next_be] = current_time + repair_duration
                
                # Add failure event
                all_events.append({
                    'Timestamp': start_time + timedelta(hours=current_time),
                    'Description': be_descriptions[next_be],
                    'Status': 'Active',
                    'Basic_Event': next_be,
                    'System_Failure': False  # Will be determined later
                })
        
        elif next_event_type == 'repair_complete':
            # BE repair completes
            if be_states[next_be] == 'failed':  # Double-check it's still failed
                be_states[next_be] = 'operational'
                
                # Add repair completion event
                all_events.append({
                    'Timestamp': start_time + timedelta(hours=current_time),
                    'Description': be_descriptions[next_be],
                    'Status': 'Cleared',
                    'Basic_Event': next_be,
                    'System_Failure': False  # Will be determined later
                })
    
    # Sort all events by timestamp
    all_events.sort(key=lambda x: x['Timestamp'])
    
    # Process events chronologically to determine system failure
    active_bes = set()
    fault_logs = []
    
    for event in all_events:
        be = event['Basic_Event']
        
        if event['Status'] == 'Active':
            active_bes.add(be)
        else:  # 'Cleared'
            active_bes.discard(be)
        
        # Convert active BEs to the same format as minimal cut sets
        active_bes_formatted = {be[2:] for be in active_bes}
        
        # Check system failure based on formatted active BEs
        system_failed = False
        for cut_set in minimal_cut_sets:
            if cut_set.issubset(active_bes_formatted):
                system_failed = True
                break
        
        # Update event with current system failure status
        event['System_Failure'] = system_failed
        fault_logs.append(event)
    
    fault_logs_df = pd.DataFrame(fault_logs)
    
    # Print statistics
    print("\nFault Log Statistics:")
    for be in sorted(failure_rates.keys()):
        active_events = fault_logs_df[(fault_logs_df['Basic_Event'] == be) & 
                                    (fault_logs_df['Status'] == 'Active')]
        cleared_events = fault_logs_df[(fault_logs_df['Basic_Event'] == be) & 
                                     (fault_logs_df['Status'] == 'Cleared')]
        print(f"  {be}: {len(active_events)} failures, {len(cleared_events)} repairs")
    
    return fault_logs_df, be_descriptions, accelerated_rates

def check_system_failure(active_bes, minimal_cut_sets):
    """
    Check if current active BEs cause system failure based on minimal cut sets
    """
    # Convert active BEs to same format as minimal cut sets
    active_bes_formatted = {be[2:] for be in active_bes}  # Remove 'BE' prefix
    
    for cut_set in minimal_cut_sets:
        if cut_set.issubset(active_bes_formatted):
            return True
    return False

def fault_logs_to_truth_table(fault_logs, be_descriptions, minimal_cut_sets):
    """
    Convert fault logs to truth table format for DDFTA algorithm
    """
    
    # Get all unique timestamps where any BE changes state
    change_timestamps = set(fault_logs['Timestamp'])
    
    truth_table_rows = []
    processed_states = set()
    
    # Sort timestamps chronologically
    sorted_timestamps = sorted(change_timestamps)
    
    for timestamp in sorted_timestamps:
        # Get state at this exact timestamp
        events_up_to_now = fault_logs[fault_logs['Timestamp'] <= timestamp]
        
        # Determine active BEs at this timestamp
        active_bes_snapshot = set()
        for _, event_row in events_up_to_now.iterrows():
            if event_row['Status'] == 'Active':
                active_bes_snapshot.add(event_row['Basic_Event'])
            else:  # 'Cleared'
                active_bes_snapshot.discard(event_row['Basic_Event'])
        
        # FIX: Use the same format conversion as in fault log generation
        active_bes_formatted = {be[2:] for be in active_bes_snapshot}
        
        # Determine system failure based on formatted active BEs
        system_failure = False
        for cut_set in minimal_cut_sets:
            if cut_set.issubset(active_bes_formatted):
                system_failure = True
                break
        
        # Create state signature for deduplication
        state_signature = (tuple(sorted(active_bes_snapshot)), system_failure)
        
        # Only add if this is a new state
        if state_signature not in processed_states:
            processed_states.add(state_signature)
            
            # Create binary row for truth table
            row_data = {}
            for be_num in range(1, 19):
                be_name = f"BE{be_num}"
                row_data[be_name] = 1 if be_name in active_bes_snapshot else 0
            
            row_data['TE'] = 1 if system_failure else 0
            
            truth_table_rows.append(row_data)
    
    truth_table = pd.DataFrame(truth_table_rows)
    
    # Ensure proper column order
    be_columns = [f"BE{i}" for i in range(1, 19)]
    truth_table = truth_table[be_columns + ['TE']]
    
    print(f"Created truth table with {len(truth_table)} rows")
    print(f"System failure states: {truth_table['TE'].sum()} out of {len(truth_table)}")
    
    return truth_table

# def fault_logs_to_truth_table(fault_logs, be_descriptions):
#     """
#     Convert fault logs to truth table format for DDFTA algorithm
#     - Creates a proper truth table with binary columns for each BE and TE
#     """
    
#     # Get all unique timestamps where any BE changes state
#     change_timestamps = set(fault_logs['Timestamp'])
    
#     truth_table_rows = []
#     processed_states = set()
    
#     # Sort timestamps chronologically
#     sorted_timestamps = sorted(change_timestamps)
    
#     for timestamp in sorted_timestamps:
#         # Get state at this exact timestamp
#         events_up_to_now = fault_logs[fault_logs['Timestamp'] <= timestamp]
        
#         # Determine active BEs at this timestamp
#         active_bes_snapshot = set()
#         for _, event_row in events_up_to_now.iterrows():
#             if event_row['Status'] == 'Active':
#                 active_bes_snapshot.add(event_row['Basic_Event'])
#             else:  # 'Cleared'
#                 active_bes_snapshot.discard(event_row['Basic_Event'])
        
#         # Get system failure status at this timestamp
#         current_events = fault_logs[fault_logs['Timestamp'] == timestamp]
#         if not current_events.empty:
#             system_failure = current_events['System_Failure'].iloc[-1]
#         else:
#             system_failure = False
        
#         # Create state signature for deduplication
#         state_signature = (tuple(sorted(active_bes_snapshot)), system_failure)
        
#         # Only add if this is a new state
#         if state_signature not in processed_states:
#             processed_states.add(state_signature)
            
#             # Create binary row for truth table - use proper BE names in order
#             row_data = {}
#             # Sort BEs numerically: BE1, BE2, BE3, ..., BE18
#             for be_num in range(1, 19):
#                 be_name = f"BE{be_num}"
#                 row_data[be_name] = 1 if be_name in active_bes_snapshot else 0
            
#             row_data['TE'] = 1 if system_failure else 0
            
#             truth_table_rows.append(row_data)
    
#     truth_table = pd.DataFrame(truth_table_rows)
    
#     # Ensure proper column order: BE1, BE2, ..., BE18, TE
#     be_columns = [f"BE{i}" for i in range(1, 19)]
#     truth_table = truth_table[be_columns + ['TE']]
    
#     print(f"Created truth table with {len(truth_table)} rows and columns: {truth_table.columns.tolist()}")
    
#     return truth_table


# def fault_logs_to_truth_table(fault_logs, be_descriptions, minimal_cut_sets):
#     """
#     Create truth table from properly generated fault logs
#     """
#     # Get unique timestamps of state changes
#     change_timestamps = sorted(set(fault_logs['Timestamp']))
    
#     truth_table_rows = []
#     processed_states = set()
    
#     for timestamp in change_timestamps:
#         # Build active BEs snapshot at this exact timestamp
#         events_up_to_now = fault_logs[fault_logs['Timestamp'] <= timestamp]
#         active_bes_snapshot = set()
        
#         for _, event_row in events_up_to_now.iterrows():
#             if event_row['Status'] == 'Active':
#                 active_bes_snapshot.add(event_row['Basic_Event'])
#             else:
#                 active_bes_snapshot.discard(event_row['Basic_Event'])
        
#         # Get the CORRECT system failure status for this timestamp
#         # Use the last event at this timestamp that has the proper system failure
#         current_events = fault_logs[fault_logs['Timestamp'] == timestamp]
#         if not current_events.empty:
#             # Use the system failure from the last event at this timestamp
#             system_failure = current_events['System_Failure'].iloc[-1]
#         else:
#             system_failure = False
        
#         # Verify consistency (for debugging)
#         calculated_failure = any(
#             all(be in active_bes_snapshot for be in cut_set) 
#             for cut_set in minimal_cut_sets
#         )
        
#         if system_failure != calculated_failure:
#             print(f"WARNING: Inconsistency at {timestamp}")
#             print(f"  Active: {sorted(active_bes_snapshot)}")
#             print(f"  Recorded TE: {system_failure}, Calculated TE: {calculated_failure}")
        
#         # Create state signature
#         state_signature = (tuple(sorted(active_bes_snapshot)), system_failure)
        
#         if state_signature not in processed_states:
#             processed_states.add(state_signature)
            
#             # Create truth table row
#             row_data = {}
#             for be_num in range(1, 19):
#                 be_name = f"BE{be_num}"
#                 row_data[be_name] = 1 if be_name in active_bes_snapshot else 0
#             row_data['TE'] = 1 if system_failure else 0
            
#             truth_table_rows.append(row_data)
    
#     truth_table = pd.DataFrame(truth_table_rows)
#     be_columns = [f"BE{i}" for i in range(1, 19)]
#     truth_table = truth_table[be_columns + ['TE']]
    
#     return truth_table