#!/usr/bin/env python3
"""
Refactored MIA evaluation script for generating comprehensive MIA results.

This script has been organized into separate utility modules for better maintainability:
- data_utils.py: Data preprocessing and file loading utilities
- evaluator_utils.py: MIA evaluation and ensemble methods  
- processing_utils.py: Main processing functions and configuration constants

Usage:
    python christy_eval_gen_mia_refactored.py
"""

import os
import argparse
from utils import (
    process_and_evaluate, 
    save_mia_results_to_csv,
    # DATASETS, 
    # SEEDS, 
    METHODS,
    SIZES,
    ATTACK_SIZES
)


def main(dataset, seed):
    """Main execution function"""
    # Configuration
    base_directory = "ensemble_data/"  # Replace with the correct base path
    output_file = 'results/mia_results.csv'
    model = 'NN'  # Fixed model for this evaluation
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    # ratio_list = [0.01,0.05, 0.1, 0.2,0.3]
    # Process all combinations of datasets, seeds, and methods
    # for dataset in DATASETS:
    print(f"\nProcessing dataset: {dataset}")
    # for seed in SEEDS:
    print(f"Using seed: {seed}")
    for method in METHODS:
        print(f"Using method: {method}")
        for Synth_size in SIZES:
            for attack_size in ATTACK_SIZES:
                # ratio = attack_size / Synth_size
                # if round(ratio, 1) not in ratio_list:
                    # continue
                try:
                    # Load numerical features if dataset is 'synth'
                    results = process_and_evaluate(
                        base_directory, dataset, seed, method ,model, attack_size ,Synth_size)
                    save_mia_results_to_csv(*results, Synth_size, attack_size=attack_size , output_file=output_file)
                    print(f"Successfully processed {dataset} with {method}, attack size {attack_size} and synth size {Synth_size}")
                    
                except Exception as e:
                    print(f"Error processing {dataset} with {method}, attack size {attack_size} and synth size {Synth_size}: {str(e)}")
        # Save individual results dataframe for later analysis
        
        
                        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset_name', type=str, default='adult', help='hospital,adult,informs,synth_adult,synth_informs,synth_hospital,compas,default_credit')
    parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 5')

    args = parser.parse_args()
    main(args.dataset_name, args.rseed)
