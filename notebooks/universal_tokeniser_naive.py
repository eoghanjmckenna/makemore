#!/usr/bin/env python3

"""
Universal Tokenizer Utility

Usage Scenarios:
----------------
1) Create new tokenizers, optionally save them, and optionally tokenize:
   python universal_tokenizer.py \
       --data_folder /path/to/data \
       --create_tokenizers \
       --max_households_create 10 \
       --num_bins 256 \
       --save_tokenizers /path/to/tokenizers.pkl \
       --do_tokenize \
       --max_households_tokenize 10 \
       --output_folder /path/to/tokenized_files \
       --seed_create 42 \
       --seed_tokenize 99

2) Load existing tokenizers and tokenize:
   python universal_tokenizer.py \
       --data_folder /path/to/data \
       --load_tokenizers /path/to/tokenizers.pkl \
       --do_tokenize \
       --max_households_tokenize 10 \
       --output_folder /path/to/tokenized_files \
       --seed_tokenize 99

3) Load as a module in Python:
   import universal_tokenizer as ut
   elec_boundaries, gas_boundaries = ut.build_tokenizers_for_files([...])
   ...
"""

import os
import glob
import argparse
import random
import pickle
from collections import Counter
from bisect import bisect_left

import numpy as np
import pandas as pd


###############################################################################
# 1. Core Functions for Building & Tokenizing
###############################################################################

def build_quantile_boundaries(data_iterable, num_bins=256):
    """
    Build quantile boundaries by:
      1. Counting unique values + frequencies (values rounded to nearest 0.1 Wh)
      2. Sorting by numeric value
      3. Computing cumulative distribution (CDF)
      4. Determining each quantile cutoff

    Returns: A list of bin edges (length = num_bins - 1).
    """
    counts = Counter()
    for val in data_iterable:
        if not pd.isna(val):
            # Round to nearest 0.1 Wh to reduce float noise
            rounded_val = round(val, 1)
            counts[rounded_val] += 1

    if not counts:
        return []

    # Sort by numeric value
    unique_items = sorted(counts.items(), key=lambda x: x[0])  # [(value, freq), ...]
    unique_values = [item[0] for item in unique_items]
    freqs = [item[1] for item in unique_items]

    # Build CDF
    cdf = []
    running_sum = 0
    for f in freqs:
        running_sum += f
        cdf.append(running_sum)
    total_count = cdf[-1]

    # Determine boundary for each quantile
    boundaries = []
    for i in range(1, num_bins):
        frac = i / num_bins
        threshold = frac * total_count
        idx = bisect_left(cdf, threshold)
        boundaries.append(unique_values[idx])

    return boundaries


def build_tokenizers_for_files(file_paths, num_bins=256):
    """
    Aggregates all 'Clean_elec_imp_hh_Wh' and 'Clean_gas_hh_Wh' from the
    given file paths, builds two sets of boundaries (elec & gas).

    Returns: (elec_boundaries, gas_boundaries)
    """
    elec_values = []
    gas_values = []

    for fp in file_paths:
        df = pd.read_pickle(fp)
        elec_values.extend(df['Clean_elec_imp_hh_Wh'].values)
        gas_values.extend(df['Clean_gas_hh_Wh'].values)

    print(f"[BUILD] Found {len(elec_values)} electric values and "
          f"{len(gas_values)} gas values in {len(file_paths)} files.")

    elec_boundaries = build_quantile_boundaries(elec_values, num_bins=num_bins)
    gas_boundaries = build_quantile_boundaries(gas_values, num_bins=num_bins)

    return elec_boundaries, gas_boundaries


def get_token(value, boundaries, missing_token, error_token,
              min_val=None, max_val=None):
    """
    Convert a numeric value into a token index based on quantile boundaries.
    Rounds the value to 0.1 Wh for consistency.
    """
    if pd.isna(value):
        return missing_token

    if len(boundaries) == 0:  # No data => treat as error
        return error_token

    rounded_val = round(value, 1)

    low_bound = boundaries[0] if min_val is None else min_val
    high_bound = boundaries[-1] if max_val is None else max_val

    if rounded_val < low_bound or rounded_val > high_bound:
        return error_token

    # Use bisect to find bin
    from bisect import bisect_left
    idx = bisect_left(boundaries, rounded_val)
    return idx


def tokenize_files(file_paths, elec_boundaries, gas_boundaries,
                   output_folder=None):
    """
    Applies the tokenizers to each file in file_paths. Optionally saves output.
    """
    # Separate token definitions for electricity and gas
    elec_usage_bins = len(elec_boundaries) + 1
    elec_missing_token = elec_usage_bins
    elec_error_token = elec_usage_bins + 1

    gas_usage_bins = len(gas_boundaries) + 1
    gas_missing_token = gas_usage_bins
    gas_error_token = gas_usage_bins + 1

    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for fp in file_paths:
        df = pd.read_pickle(fp)

        df['Elec_Token'] = df['Clean_elec_imp_hh_Wh'].apply(
            lambda x: get_token(
                x, elec_boundaries,
                elec_missing_token, elec_error_token
            )
        )
        df['Gas_Token'] = df['Clean_gas_hh_Wh'].apply(
            lambda x: get_token(
                x, gas_boundaries,
                gas_missing_token, gas_error_token
            )
        )

        if output_folder:
            base_name = os.path.basename(fp)
            out_fp = os.path.join(output_folder,
                                  base_name.replace(".pkl", "_tokenized.pkl"))
            df.to_pickle(out_fp)

        print(f"[TOKENIZE] Processed file: {fp}")


###############################################################################
# 2. Saving & Loading the Boundaries
###############################################################################

def save_tokenizers(elec_boundaries, gas_boundaries, out_path):
    """Save boundaries to a pickle file."""
    data = {
        "elec_boundaries": elec_boundaries,
        "gas_boundaries": gas_boundaries
    }
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[SAVE] Tokenizers saved to {out_path}")


def load_tokenizers(in_path):
    """Load boundaries from a pickle file."""
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    elec_boundaries = data["elec_boundaries"]
    gas_boundaries = data["gas_boundaries"]
    print(f"[LOAD] Tokenizers loaded from {in_path}")
    return elec_boundaries, gas_boundaries


###############################################################################
# 3. Utility Functions for Gathering File Paths with Optional Random Sampling
###############################################################################

def get_file_paths(folder_path, max_households=None, seed=None):
    """
    Returns a list of .pkl file paths in `folder_path`.
    If max_households < total number of files, randomly sample that many.
    """
    files = glob.glob(os.path.join(folder_path, "*.pkl"))
    print(f"[FILES] Found {len(files)} pkl files in {folder_path}.")

    if seed is not None:
        random.seed(seed)

    if max_households is not None and max_households < len(files):
        files = random.sample(files, max_households)
        print(f"[FILES] Randomly selected {len(files)} out of {len(files)} with seed={seed}.")

    return files


###############################################################################
# 4. Main (CLI) - Putting It All Together
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Universal Tokenizer Utility")
    # Flags for optional actions
    parser.add_argument("--create_tokenizers", action="store_true",
                        help="If set, build new tokenizers from data.")
    parser.add_argument("--do_tokenize", action="store_true",
                        help="If set, tokenize the data using the tokenizers (created or loaded).")

    # Folder for data
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Folder containing .pkl files.")

    # Options for building tokenizers
    parser.add_argument("--max_households_create", type=int, default=None,
                        help="If creating tokenizers, randomly sample this many files.")
    parser.add_argument("--seed_create", type=int, default=None,
                        help="Random seed for selecting files for building tokenizers.")
    parser.add_argument("--num_bins", type=int, default=256,
                        help="Number of bins for quantile splitting when creating tokenizers.")

    # Options for loading tokenizers
    parser.add_argument("--load_tokenizers", type=str, default=None,
                        help="Path to an existing .pkl tokenizers file to load (skip creation).")

    # Options for saving tokenizers
    parser.add_argument("--save_tokenizers", type=str, default=None,
                        help="Path to save created tokenizers (if create_tokenizers is true).")

    # Options for tokenizing
    parser.add_argument("--max_households_tokenize", type=int, default=None,
                        help="If tokenizing, randomly sample this many files.")
    parser.add_argument("--seed_tokenize", type=int, default=None,
                        help="Random seed for selecting files for tokenization.")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Folder to write tokenized .pkl files.")

    args = parser.parse_args()

    # 1. Decide how to get the tokenizers
    elec_boundaries, gas_boundaries = None, None

    if args.load_tokenizers:
        # If user provided a path to load from, do that first
        elec_boundaries, gas_boundaries = load_tokenizers(args.load_tokenizers)
    elif args.create_tokenizers:
        # If user wants to create them
        file_paths_create = get_file_paths(
            folder_path=args.data_folder,
            max_households=args.max_households_create,
            seed=args.seed_create
        )
        elec_boundaries, gas_boundaries = build_tokenizers_for_files(
            file_paths=file_paths_create,
            num_bins=args.num_bins
        )
        # Optionally save
        if args.save_tokenizers:
            save_tokenizers(elec_boundaries, gas_boundaries, args.save_tokenizers)

    # If we still don't have boundaries, and user wants to tokenize => problem
    if args.do_tokenize and (elec_boundaries is None or gas_boundaries is None):
        raise ValueError("Cannot tokenize without tokenizers. "
                         "Either load them or create them first.")

    # 2. Tokenize if requested
    if args.do_tokenize:
        file_paths_tokenize = get_file_paths(
            folder_path=args.data_folder,
            max_households=args.max_households_tokenize,
            seed=args.seed_tokenize
        )
        tokenize_files(
            file_paths=file_paths_tokenize,
            elec_boundaries=elec_boundaries,
            gas_boundaries=gas_boundaries,
            output_folder=args.output_folder
        )

    print("[DONE] All requested operations completed.")


###############################################################################
# 5. If Running as a Script
###############################################################################

if __name__ == "__main__":
    main()
