#!/usr/bin/env python3

import os
import glob
import argparse

import numpy as np
import pandas as pd


###############################################################################
# 1. Helper Functions: Naive Quantile Binning
###############################################################################

def build_quantile_boundaries(data_iterable, num_bins=256):
    """
    Construct quantile boundaries using a naive sort-based approach.
    WARNING: This requires loading the entire dataset into memory.

    Args:
      data_iterable: An iterable of numeric values (e.g., a list or NumPy array).
      num_bins (int): Desired number of bins.

    Returns:
      boundaries (list of floats): A sorted list of bin edges (length = num_bins - 1).
        For num_bins=256, you'll get 255 boundary values.
    """
    # Convert to a NumPy array, remove any NaNs for boundary calculation
    data_array = np.array([v for v in data_iterable if not pd.isna(v)], dtype=float)
    n = len(data_array)
    if n == 0:
        # Edge case: if data is empty, return an empty boundary list
        return []

    # Sort the entire array (O(N log N))
    data_array.sort()

    boundaries = []
    # For example, with 256 bins, we compute boundaries at 1/256, 2/256, ..., 255/256 quantiles
    for i in range(1, num_bins):
        # Index in the sorted array
        q_idx = int(np.floor(i * n / num_bins))
        if q_idx >= n:
            q_idx = n - 1
        boundaries.append(data_array[q_idx])
    return boundaries

###############################################################################
# 2. Tokenization Functions
###############################################################################

def get_token(value, boundaries, missing_token, error_token, min_val=None, max_val=None):
    """
    Convert a numeric `value` into a discrete token index based on `boundaries`.

    Args:
      value: The numeric consumption value.
      boundaries: List of sorted boundary values that define the bin edges.
      missing_token: The integer token representing missing data.
      error_token: The integer token representing out-of-bounds/error data.
      min_val: Lower plausible bound for valid readings (None => use boundaries[0]).
      max_val: Upper plausible bound for valid readings (None => use boundaries[-1]).

    Returns:
      An integer token index [0..len(boundaries)] or a special missing/error token.
    """
    # Handle missing
    if pd.isna(value):
        return missing_token
    
    # If we have no boundaries (edge case with no valid data), treat as error
    if len(boundaries) == 0:
        return error_token

    # Determine lower/upper from boundaries if none given
    lower_bound = boundaries[0] if min_val is None else min_val
    upper_bound = boundaries[-1] if max_val is None else max_val

    # Handle errors / out-of-bounds
    if value < lower_bound or value > upper_bound:
        return error_token

    # Binary search to find bin
    lo, hi = 0, len(boundaries)
    while lo < hi:
        mid = (lo + hi) // 2
        if value < boundaries[mid]:
            hi = mid
        else:
            lo = mid + 1
    # 'lo' is the bin index
    return lo

def decode_token(token_idx, boundaries, missing_token, error_token):
    """
    Inverse operation: map a token index back to a numeric estimate.
    Uses the midpoint of the bin edges as a representative value.

    Args:
      token_idx: The integer token index.
      boundaries: List of sorted boundary values.
      missing_token: Special token index for missing data.
      error_token: Special token index for errors.

    Returns:
      A float representing an approximate usage, or np.nan/None for special tokens.
    """
    if token_idx == missing_token:
        return np.nan
    if token_idx == error_token:
        return None  # or another sentinel

    num_bins = len(boundaries) + 1
    # Bins: index in [0, num_bins-1]
    # bin i covers [boundaries[i-1], boundaries[i]) with i in [1..num_bins-2],
    # plus lower edge i=0 and upper edge i=(num_bins-1).
    if token_idx == 0:
        low_val = float('-inf')
        high_val = boundaries[0]
    elif token_idx == num_bins - 1:
        low_val = boundaries[-1]
        high_val = float('inf')
    else:
        low_val = boundaries[token_idx - 1] if token_idx > 0 else boundaries[0]
        high_val = boundaries[token_idx] if token_idx < len(boundaries) else boundaries[-1]

    # Use midpoint if finite
    if np.isfinite(low_val) and np.isfinite(high_val):
        return (low_val + high_val) / 2.0
    elif not np.isfinite(low_val):
        return high_val
    else:
        return low_val

###############################################################################
# 3. Main Processing: Reading PKL Files & Building Tokenizers
###############################################################################

def build_tokenizers_for_folder(folder_path, max_households=None, num_bins=256):
    """
    Reads up to `max_households` pickled DataFrames in `folder_path`, aggregates 
    electricity and gas usage, then returns quantile boundaries for each.

    Args:
      folder_path (str): Path to the folder containing household .pkl files.
      max_households (int or None): If specified, only read that many files.
      num_bins (int): Number of bins for quantile splitting.

    Returns:
      elec_boundaries (list of floats),
      gas_boundaries (list of floats)
    """
    file_paths = glob.glob(os.path.join(folder_path, "*.pkl"))
    if max_households is not None:
        file_paths = file_paths[:max_households]

    elec_values = []
    gas_values = []

    for fp in file_paths:
        df = pd.read_pickle(fp)
        # We assume columns: datetime, halfhour_from_midnight,
        #                    Clean_elec_imp_hh_Wh, Clean_gas_hh_Wh, temp_C

        # Extend aggregator lists with valid numeric usage
        elec_values.extend(df['Clean_elec_imp_hh_Wh'].values)
        gas_values.extend(df['Clean_gas_hh_Wh'].values)

    # Build quantile boundaries for electricity
    print(f"Building electricity quantile boundaries from {len(elec_values)} records...")
    elec_boundaries = build_quantile_boundaries(elec_values, num_bins=num_bins)

    # Build quantile boundaries for gas
    print(f"Building gas quantile boundaries from {len(gas_values)} records...")
    gas_boundaries = build_quantile_boundaries(gas_values, num_bins=num_bins)

    return elec_boundaries, gas_boundaries

def tokenize_folder(folder_path, elec_boundaries, gas_boundaries, 
                    output_folder=None, max_households=None):
    """
    Tokenizes each .pkl file in `folder_path` using the provided boundaries.
    Optionally writes out a tokenized .pkl copy to `output_folder`.

    Args:
      folder_path (str): Path containing the .pkl files.
      elec_boundaries (list of floats): Electricity bin edges.
      gas_boundaries (list of floats): Gas bin edges.
      output_folder (str or None): If specified, writes tokenized DataFrames to this folder.
      max_households (int or None): Number of files to process for testing.

    Returns:
      None
    """
    # Define special token indices
    # For num_bins usage bins, total token range is [0..(num_bins-1)]
    # We'll put missing_token and error_token after that range.
    usage_bins = len(elec_boundaries) + 1  # e.g. 256
    missing_token = usage_bins
    error_token = usage_bins + 1

    file_paths = glob.glob(os.path.join(folder_path, "*.pkl"))
    if max_households is not None:
        file_paths = file_paths[:max_households]

    # Create output folder if needed
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for fp in file_paths:
        df = pd.read_pickle(fp)
        
        # Tokenize electricity
        df['Elec_Token'] = df['Clean_elec_imp_hh_Wh'].apply(
            lambda x: get_token(
                x, elec_boundaries, 
                missing_token=missing_token, 
                error_token=error_token,
                min_val=None,  # Optionally set a min
                max_val=None   # Optionally set a max
            )
        )
        
        # Tokenize gas
        df['Gas_Token'] = df['Clean_gas_hh_Wh'].apply(
            lambda x: get_token(
                x, gas_boundaries, 
                missing_token=missing_token, 
                error_token=error_token,
                min_val=None,
                max_val=None
            )
        )

        # Optionally save the tokenized data
        if output_folder:
            base_name = os.path.basename(fp)
            out_fp = os.path.join(output_folder, base_name.replace(".pkl", "_tokenized.pkl"))
            df.to_pickle(out_fp)

        print(f"Tokenized file: {fp}")

###############################################################################
# 4. Command-Line Interface
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Universal Tokenizer (Naive) for Household Data")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to folder containing household .pkl files.")
    parser.add_argument("--max_households", type=int, default=None,
                        help="If specified, process only this many files for testing.")
    parser.add_argument("--num_bins", type=int, default=256,
                        help="Number of bins for quantile splitting.")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Folder to write tokenized .pkl files. If not specified, no output is written.")
    args = parser.parse_args()

    # Step 1: Build boundaries (electricity + gas)
    elec_boundaries, gas_boundaries = build_tokenizers_for_folder(
        folder_path=args.data_folder,
        max_households=args.max_households,
        num_bins=args.num_bins
    )

    # Step 2: Tokenize each file
    tokenize_folder(
        folder_path=args.data_folder,
        elec_boundaries=elec_boundaries,
        gas_boundaries=gas_boundaries,
        output_folder=args.output_folder,
        max_households=args.max_households
    )

    print("Tokenization process completed successfully.")

if __name__ == "__main__":
    main()
