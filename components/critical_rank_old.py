
import pandas as pd
from collections import defaultdict

#%% Data

# Minimal Cut Sets from your fault tree
minimal_cut_sets = [
                    ['18'], ['17'], ['16'], ['15'], ['14'], ['13'], ['12'], ['11'], ['10'],
                    ['9'], ['8'], ['7'], ['6'], ['5'], ['2'], ['1'], ['3', '4']
                    ]
# Mapping from basic event labels to probabilities
failure_probs = {
                'BE1': 0.0001,  # Fuse_Oxidation
                'BE2': 0.0002,  # Fuse_Improper_Maintenance
                'BE3': 0.0008,  # MCB_Faulty
                'BE4': 0.0846,  # PV_Broken_Interconnect
                'BE5': 0.0490,  # PV_Grounding_System_Fault
                'BE6': 0.0003,  # PV_Glass_Breakage
                'BE7': 0.0013,  # PV_Soiling
                'BE8': 0.0088,  # PV_Shading
                'BE9': 0.1115,  # PV_Broken_Cell
                'BE10': 0.1487, # PV_Solder_Bond_Failure
                'BE11': 0.0101, # PV_Hot_Spot
                'BE12': 0.0021, # PV_Faulty_Bypass_Diode
                'BE13': 0.0052, # PV_Short_Open_Circuit
                'BE14': 0.0729, # Rack_Structure_Fault
                'BE15': 0.0570, # Encapsulant_Fault
                'BE16': 0.0001, # Cable_Insulation_Failure
                'BE17': 0.0002, # Cable_Material_Aging
                'BE18': 0.0002  # Placeholder if needed
                }

#%% Structure Importance


# Initialize dictionary to hold structure importance
importance = defaultdict(float)

for mcs in minimal_cut_sets:
    size = len(mcs)
    contrib = 1 / (2 ** (size - 1))
    for event in mcs:
        importance[event] += contrib

# Convert to sorted list
ranked_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Display
df_importance = pd.DataFrame(list(importance.items()), columns=["Basic Event", "Importance"])
df_importance.sort_values(by="Importance", ascending=False, inplace=True)
df_importance.reset_index(drop=True, inplace=True)


#%% Birnbaum Importance

# Assume 'truth_table' is a DataFrame with BE1..BE18 and TE columns
# Example: truth_table = generate_truth_table_from_expression(expression)

def compute_birnbaum_importance(truth_table, failure_probs):
    importance = {}
    be_columns = [col for col in truth_table.columns if col.startswith('BE')]
    
    for be in be_columns:
        # Fix BE to 1 (failure)
        prob_T_when_1 = truth_table[truth_table[be] == 1].copy()
        for other_be in be_columns:
            if other_be != be:
                prob_T_when_1[other_be] = failure_probs[other_be]
        prob_T1 = (prob_T_when_1.eval(' & '.join([f'{b} == {failure_probs[b]}' for b in be_columns if b != be])) * prob_T_when_1['TE']).sum()

        # Fix BE to 0 (working)
        prob_T_when_0 = truth_table[truth_table[be] == 0].copy()
        for other_be in be_columns:
            if other_be != be:
                prob_T_when_0[other_be] = failure_probs[other_be]
        prob_T0 = (prob_T_when_0.eval(' & '.join([f'{b} == {failure_probs[b]}' for b in be_columns if b != be])) * prob_T_when_0['TE']).sum()
        
        # Birnbaum Importance
        importance[be] = prob_T1 - prob_T0
    
    # Return sorted importance
    return sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Example usage:
# ranked_importance = compute_birnbaum_importance(truth_table, failure_probs)
# for be, score in ranked_importance:
#     print(f"{be}: {score:.6f}")


#%% Criticality Importance

def compute_criticality_importance(truth_table, failure_probs):
    be_columns = [col for col in truth_table.columns if col.startswith('BE')]
    
    # Step 1: Calculate total system failure probability P(T)
    P_T = 0
    for idx, row in truth_table.iterrows():
        p_row = 1
        for be in be_columns:
            prob = failure_probs[be]
            p_row *= prob if row[be] == 1 else (1 - prob)
        P_T += p_row * row['TE']
    
    # Step 2: Calculate P(T ∩ BE_i) for each basic event
    criticality = {}
    for be in be_columns:
        joint_prob = 0
        for idx, row in truth_table.iterrows():
            if row[be] == 1 and row['TE'] == 1:
                p_row = 1
                for b in be_columns:
                    prob = failure_probs[b]
                    p_row *= prob if row[b] == 1 else (1 - prob)
                joint_prob += p_row
        # IC = P(T ∩ BE_i) / P(T)
        criticality[be] = joint_prob / P_T if P_T > 0 else 0

    # Sort by importance descending
    return sorted(criticality.items(), key=lambda x: x[1], reverse=True)

# Example usage:
# criticality_importance = compute_criticality_importance(truth_table, failure_probs)
# for be, ic in criticality_importance:
#     print(f"{be}: {ic:.6f}")
