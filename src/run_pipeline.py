#!/usr/bin/env python3
"""
Orchestrator script to run the entire LLM evaluation pipeline for a given dataset.

This script automates the process of:
1. Running unified_models_v2.py to generate LLM responses for each stage
2. Running column_merger_nejm_ordered.py to merge responses and create long format data
3. Running score_evaluater_inbatches.py to evaluate LLM responses

Usage:
    python src/run_pipeline.py <dataset_folder_name>

Example:
    python src/run_pipeline.py cases_typical_concise_selected
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the entire LLM evaluation pipeline for a dataset.')
    parser.add_argument('dataset', type=str, help='Dataset folder name (e.g., cases_typical_concise_selected)')
    parser.add_argument('--stages', nargs='+', default=['stage_1', 'stage_2', 'stage_3'], 
                        help='Stages to process (default: stage_1 stage_2 stage_3)')
    parser.add_argument('--skip-unified', action='store_true', 
                        help='Skip running unified_models_v2.py (useful if LLM responses are already generated)')
    parser.add_argument('--skip-merger', action='store_true', 
                        help='Skip running column_merger_nejm_ordered.py (useful if merged files are already generated)')
    parser.add_argument('--skip-scorer', action='store_true', 
                        help='Skip running score_evaluater_inbatches.py (useful if scores are already generated)')
    return parser.parse_args()

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with exit code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"SUCCESS: {description} completed in {elapsed_time:.2f} seconds")
    print(f"OUTPUT: {result.stdout}")
    return True

def check_dataset_exists(dataset):
    """Check if the dataset folder exists."""
    if not os.path.isdir(dataset):
        print(f"ERROR: Dataset folder '{dataset}' does not exist.")
        return False
    return True

def check_stage_exists(dataset, stage):
    """Check if the stage folder exists within the dataset."""
    stage_path = os.path.join(dataset, stage)
    if not os.path.isdir(stage_path):
        print(f"ERROR: Stage folder '{stage_path}' does not exist.")
        return False
    return True

def extract_dataset_name(dataset_path):
    """Extract the dataset name from the path."""
    return os.path.basename(dataset_path)

def main():
    """Main function to run the pipeline."""
    args = parse_arguments()
    dataset = args.dataset
    stages = args.stages
    
    # Check if dataset exists
    if not check_dataset_exists(dataset):
        sys.exit(1)
    
    # Get the dataset name (without path)
    dataset_name = extract_dataset_name(dataset)
    
    # Process each stage
    for stage in stages:
        print(f"\n\n{'#'*100}")
        print(f"# PROCESSING: {dataset} - {stage}")
        print(f"{'#'*100}\n")
        
        # Check if stage exists
        if not check_stage_exists(dataset, stage):
            print(f"Skipping {stage} as it does not exist in {dataset}")
            continue
        
        # Step 1: Run unified_models_v2.py
        if not args.skip_unified:
            medical_cases_dir = f"{dataset}/{stage}"
            unified_command = [
                "python", "src/unified_models_v2.py",
                "--medical_cases_dir", medical_cases_dir
            ]
            if not run_command(unified_command, f"Generate LLM responses for {dataset}/{stage}"):
                print(f"ERROR: Failed to generate LLM responses for {dataset}/{stage}")
                continue
        else:
            print("Skipping unified_models_v2.py as requested")
        
        # Step 2: Run column_merger_nejm_ordered.py
        if not args.skip_merger:
            merger_command = [
                "python", "src/column_merger_nejm_ordered.py",
                "--dataset", dataset_name,
                "--stage", stage
            ]
            if not run_command(merger_command, f"Merge LLM responses for {dataset_name}_{stage}"):
                print(f"ERROR: Failed to merge LLM responses for {dataset_name}_{stage}")
                continue
        else:
            print("Skipping column_merger_nejm_ordered.py as requested")
        
        # Step 3: Run score_evaluater_inbatches.py
        if not args.skip_scorer:
            scorer_command = [
                "python", "src/score_evaluater_inbatches.py",
                "--dataset", dataset_name,
                "--stage", stage
            ]
            if not run_command(scorer_command, f"Score LLM responses for {dataset_name}_{stage}"):
                print(f"ERROR: Failed to score LLM responses for {dataset_name}_{stage}")
                continue
        else:
            print("Skipping score_evaluater_inbatches.py as requested")
        
        print(f"\n{'='*80}")
        print(f"COMPLETED: {dataset} - {stage}")
        print(f"{'='*80}\n")
    
    print("\n\nPIPELINE EXECUTION COMPLETE!")
    print(f"Processed dataset: {dataset}")
    print(f"Processed stages: {', '.join(stages)}")

if __name__ == "__main__":
    main()
