import pandas as pd

def validate_truth_tables(original_path, constructed_path):
    # Load the truth tables
    original = pd.read_csv(original_path, sep=" ")
    constructed = pd.read_csv(constructed_path, sep=" ")

    # Ensure consistent column ordering
    cols = sorted([col for col in constructed.columns if col != 'TE'], key=lambda x: int(x)) + ['TE']
    constructed = constructed[cols]
    original = original[cols]

    # Convert rows to sets of tuples for fast lookup
    original_rows_set = set(tuple(row) for row in original.to_numpy())

    # Find unmatched constructed rows
    unmatched_indices = []

    for idx, row in enumerate(constructed.to_numpy()):
        if tuple(row) not in original_rows_set:
            unmatched_indices.append(idx)

    # Result
    if not unmatched_indices:
        print("\n[Validation Successful]: Truth Tables Match!\n\nThe constructed fault tree produces an identical truth table to the original.")
    else:
        print(f"\n[Validation Warning]: {len(unmatched_indices)} unmatched rows found.")
        mismatch = constructed.iloc[unmatched_indices]
        print("\nFirst few unmatched constructed rows:\n")
        print(mismatch.head())
