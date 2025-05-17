#!/usr/bin/env python3
"""
Script to randomly select 25 cases from the cases_typical_concise folder
and copy them to a new folder structure (cases_typical_concise_selected).
The same cases are selected across all stages (stage_1, stage_2, stage_3).
Also copies the case list CSV file to the new directory.
"""

import os
import random
import shutil
import sys
import re
import pandas as pd

# Configuration
# SOURCE_DIR = "cases_typical_concise" 
# TARGET_DIR = "cases_typical_concise_selected"
# STAGES = ["stage_1", "stage_2", "stage_3"]
# NUM_CASES_TO_SELECT = 40
# CASE_LIST_CSV = "cases_typical_case_list.csv"

SOURCE_DIR = "cases_CPS_concise" 
TARGET_DIR = "cases_CPS_concise_selected"
STAGES = ["stage_1", "stage_2", "stage_3"]
NUM_CASES_TO_SELECT = 50
CASE_LIST_CSV = "cases_CPS_case_list.csv"

def main():
    # Check if target directory already exists
    if os.path.exists(TARGET_DIR):
        print(f"Error: Target directory '{TARGET_DIR}' already exists.")
        print("Please remove or rename it before running this script.")
        sys.exit(1)
    
    # Create the main target directory
    os.makedirs(TARGET_DIR, exist_ok=False)
    print(f"Created directory: {TARGET_DIR}")
    
    # Get all available case numbers from stage_1 (assuming all stages have the same cases)
    source_stage_dir = os.path.join(SOURCE_DIR, STAGES[0])
    if not os.path.exists(source_stage_dir):
        print(f"Error: Source directory '{source_stage_dir}' does not exist.")
        sys.exit(1)
    
    # Extract case numbers from filenames (e.g., "1.txt" -> 1)
    all_files = [f for f in os.listdir(source_stage_dir) if f.endswith('.txt')]
    case_numbers = []
    for file in all_files:
        match = re.match(r'(\d+)\.txt', file)
        if match:
            case_numbers.append(int(match.group(1)))
    
    # Check if there are enough cases
    if len(case_numbers) < NUM_CASES_TO_SELECT:
        print(f"Warning: Only {len(case_numbers)} cases found, " 
              f"but {NUM_CASES_TO_SELECT} were requested. Using all available cases.")
        selected_case_numbers = case_numbers
    else:
        # Randomly select case numbers
        selected_case_numbers = sorted(random.sample(case_numbers, NUM_CASES_TO_SELECT))
    
    print(f"Selected case numbers: {selected_case_numbers}")
    
    # Process each stage using the same selected case numbers
    for stage in STAGES:
        process_stage(stage, selected_case_numbers)
    
    # Copy and filter the case list CSV file
    copy_and_filter_case_list(selected_case_numbers)
    
    print(f"\nDone! Selected cases have been copied to '{TARGET_DIR}'")

def copy_and_filter_case_list(selected_case_numbers):
    """Copy the case list CSV file to the target directory, filtering for only the selected cases."""
    source_csv = os.path.join(SOURCE_DIR, CASE_LIST_CSV)
    target_csv = os.path.join(TARGET_DIR, CASE_LIST_CSV)
    
    if not os.path.exists(source_csv):
        print(f"Warning: Case list CSV file '{source_csv}' does not exist. Skipping.")
        return
    
    try:
        # Read the original CSV
        df = pd.read_csv(source_csv)
        
        # Filter for only the selected case numbers
        # Assuming the case number is in a column named 'case_number' or similar
        # Adjust the column name if needed
        case_number_col = None
        for col in df.columns:
            if 'case' in col.lower() and 'number' in col.lower():
                case_number_col = col
                break
        
        if case_number_col:
            filtered_df = df[df[case_number_col].isin(selected_case_numbers)]
            filtered_df.to_csv(target_csv, index=False)
            print(f"Filtered case list CSV saved to '{target_csv}'")
        else:
            # If we can't find a case number column, just copy the whole file
            shutil.copy2(source_csv, target_csv)
            print(f"Case list CSV copied to '{target_csv}' (not filtered)")
    
    except Exception as e:
        print(f"Error processing case list CSV: {e}")
        # Fall back to simple file copy if pandas fails
        shutil.copy2(source_csv, target_csv)
        print(f"Case list CSV copied to '{target_csv}' (not filtered)")

def process_stage(stage, selected_case_numbers):
    """Process a single stage by copying the selected case files."""
    source_stage_dir = os.path.join(SOURCE_DIR, stage)
    target_stage_dir = os.path.join(TARGET_DIR, stage)
    
    # Check if source stage directory exists
    if not os.path.exists(source_stage_dir):
        print(f"Warning: Source directory '{source_stage_dir}' does not exist. Skipping.")
        return
    
    # Create the target stage directory
    os.makedirs(target_stage_dir, exist_ok=True)
    print(f"Created directory: {target_stage_dir}")
    
    # Copy selected files to target directory
    selected_files = [f"{case_num}.txt" for case_num in selected_case_numbers]
    copied_files = []
    
    for file in selected_files:
        source_file = os.path.join(source_stage_dir, file)
        target_file = os.path.join(target_stage_dir, file)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            copied_files.append(file)
        else:
            print(f"Warning: File '{source_file}' does not exist and will be skipped.")
    
    print(f"Copied {len(copied_files)} files from '{source_stage_dir}' to '{target_stage_dir}'")

if __name__ == "__main__":
    main()
