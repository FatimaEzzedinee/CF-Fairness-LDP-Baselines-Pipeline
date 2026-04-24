import numpy as np
import os
import sys
import gc
import signal
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import pandas as pd
from utils_generate import *

# Define seeds and models
models = {
    #"bn": train_sample_bn,
    "ddpm": train_sample_ddpm,
    "arf": train_sample_arf,
    "tvae": train_sample_tvae,
    "ctgan": train_sample_ctgan,
    "nflow": train_sample_nflows,
    "adsgan": train_sample_adsgan,
    "pategan": train_sample_pategan,
}

# Data split seed (fixed for consistent splits)
data_split_seed = 42
# Model training/generation seeds
model_seeds = [1,2,3,4,5]

# Synthetic data generation multipliers
synth_multipliers = [1]  

def timeout_model_training(model_func, mem_set, model_seed, timeout_minutes=45):
    """Train model with timeout using threading"""
    result = {'model': None, 'error': None}

    def target_function():
        try:
            synth_model = model_func(mem_set, random_state=model_seed)
            result['model'] = synth_model
        except Exception as e:
            result['error'] = str(e)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(target_function)
        try:
            future.result(timeout=timeout_minutes * 60)  # Convert minutes to seconds
            if result['error']:
                return None, result['error']
            return result['model'], "success"
        except TimeoutError:
            return None, "timeout"
        except Exception as e:
            return None, str(e)
def clean_workspace(workspace_dir='./workspace'):
    """
    Thoroughly clean the workspace directory
    """
    if not os.path.exists(workspace_dir):
        print(f"      Workspace directory {workspace_dir} does not exist")
        return
    
    try:
        # Method 1: Use shutil.rmtree for complete removal
        shutil.rmtree(workspace_dir)
        print(f"      Successfully removed entire workspace directory: {workspace_dir}")
        
        # Recreate the empty directory if needed
        os.makedirs(workspace_dir, exist_ok=True)
        print(f"      Recreated empty workspace directory: {workspace_dir}")
        
    except Exception as e:
        print(f"      Failed to remove workspace with shutil.rmtree: {e}")
        
        # Method 2: Fallback to manual deletion with better error handling
        try:
            files_deleted = 0
            dirs_deleted = 0
            
            for root, dirs, files in os.walk(workspace_dir, topdown=False):
                # Delete all files first
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    try:
                        os.chmod(file_path, 0o777)  # Ensure we have write permissions
                        os.remove(file_path)
                        files_deleted += 1
                    except Exception as file_error:
                        print(f"        Failed to delete file {file_path}: {file_error}")
                
                # Then delete directories (bottom-up due to topdown=False)
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.chmod(dir_path, 0o777)  # Ensure we have write permissions
                        os.rmdir(dir_path)
                        dirs_deleted += 1
                    except Exception as dir_error:
                        print(f"        Failed to delete directory {dir_path}: {dir_error}")
            
            print(f"      Manual cleanup: deleted {files_deleted} files and {dirs_deleted} directories")
            
            # Try to remove the root workspace directory itself
            try:
                if os.path.exists(workspace_dir):
                    os.rmdir(workspace_dir)
                    print(f"      Removed root workspace directory")
                    # Recreate it empty
                    os.makedirs(workspace_dir, exist_ok=True)
            except Exception as root_error:
                print(f"      Could not remove root workspace directory: {root_error}")
                
        except Exception as fallback_error:
            print(f"      Fallback cleanup also failed: {fallback_error}")

# Replace the existing cleanup section with this:
def cleanup_workspace_after_model(model_name, model_seed):
    """Clean workspace after each model run"""
    print(f"      Cleaning workspace after model {model_name}, seed {model_seed}")
    clean_workspace('./workspace')
    
    # Also check for other common temporary directories that might be created
    temp_dirs = ['./temp', './tmp', './__pycache__', './logs']
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"      Also cleaned temporary directory: {temp_dir}")
            except Exception as e:
                print(f"      Failed to clean {temp_dir}: {e}")

def diagnose_workspace(workspace_dir='./workspace'):
    """Diagnose what's in the workspace directory"""
    if not os.path.exists(workspace_dir):
        print(f"      Workspace directory {workspace_dir} does not exist")
        return
    
    print(f"      Diagnosing workspace directory: {workspace_dir}")
    total_files = 0
    total_dirs = 0
    
    try:
        for root, dirs, files in os.walk(workspace_dir):
            level = root.replace(workspace_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"      {indent}{os.path.basename(root)}/")
            
            subindent = ' ' * 2 * (level + 1)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    file_size = os.path.getsize(file_path)
                    print(f"      {subindent}{file_name} ({file_size} bytes)")
                    total_files += 1
                except Exception as e:
                    print(f"      {subindent}{file_name} (error: {e})")
                    total_files += 1
            
            total_dirs += len(dirs)
    
        print(f"      Total: {total_files} files, {total_dirs} directories")
        
    except Exception as e:
        print(f"      Error diagnosing workspace: {e}")
# Get all dataset files from the data/ folder
data_folder = 'data/'
data_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])

for data_file in data_files:
    try:
        print(f"Processing dataset: {data_file}")
        df = pd.read_csv(os.path.join(data_folder, data_file))
        df = df.dropna()

        if len(df) < 10:
            print(f"Dataset {data_file} too small ({len(df)} rows), skipping...")
            continue

        dataset_name = os.path.splitext(data_file)[0]
        dataset_folder = os.path.join('ensemble_data', dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        print(f"Splitting data for {dataset_name}...")
        np.random.seed(data_split_seed)
        shuffled_indices = np.random.permutation(len(df))

        train_size = int(0.8 * len(df))
        holdout_size = int(0.1 * len(df))
        ref_size = len(df) - train_size - holdout_size

        mem_set = df.iloc[shuffled_indices[:train_size]]
        holdout_set = df.iloc[shuffled_indices[train_size:train_size + holdout_size]]
        ref_set = df.iloc[shuffled_indices[train_size + holdout_size:]]

        print(f"Data split - Member: {len(mem_set)}, Holdout: {len(holdout_set)}, Reference: {len(ref_set)}")

        mem_set.to_csv(os.path.join(dataset_folder, 'mem_set.csv'), index=False)
        holdout_set.to_csv(os.path.join(dataset_folder, 'holdout_set.csv'), index=False)
        ref_set.to_csv(os.path.join(dataset_folder, 'ref_set.csv'), index=False)

        for model_name, model_func in models.items():
            print(f"\nProcessing model: {model_name}")
            model_folder = os.path.join(dataset_folder, model_name)
            os.makedirs(model_folder, exist_ok=True)

            for model_seed in model_seeds:
                print(f"  Using model seed: {model_seed}")
                seed_folder = os.path.join(model_folder, f'seed_{model_seed}')

                if os.path.exists(seed_folder):
                    all_completed = True
                    for multiplier in synth_multipliers:
                        synth_file = os.path.join(seed_folder, f'synth_{multiplier}x.csv')
                        if not os.path.exists(synth_file):
                            all_completed = False
                            break
                    if all_completed:
                        print(f"    All synthetic data already exists for seed {model_seed}, skipping...")
                        continue

                os.makedirs(seed_folder, exist_ok=True)

                try:
                    np.random.seed(model_seed)
                    print(f"    Training model with timeout (30 mins)...")
                    start_time = time.time()

                    synth_model, status = timeout_model_training(model_func, mem_set, model_seed, timeout_minutes=360)
                    training_time = time.time() - start_time

                    if status == "timeout":
                        print(f"    Model {model_name} timed out after 30 minutes, skipping...")
                        with open(os.path.join(seed_folder, 'timeout_marker.txt'), 'w') as f:
                            f.write(f"Model training timed out after 30 minutes\nSeed: {model_seed}\n")
                        continue
                    elif status != "success":
                        print(f"    Model {model_name} failed to train: {status}")
                        with open(os.path.join(seed_folder, 'error_marker.txt'), 'w') as f:
                            f.write(f"Model training failed: {status}\nSeed: {model_seed}\n")
                        continue

                    print(f"    Model trained successfully in {training_time:.2f} seconds")

                    for multiplier in synth_multipliers:
                        synth_file = os.path.join(seed_folder, f'synth_{multiplier}x.csv')
                        if os.path.exists(synth_file):
                            print(f"    Synthetic data {multiplier}x already exists, skipping...")
                            continue
                        try:
                            print(f"    Generating {multiplier}x synthetic data...")
                            synth_size = len(mem_set) * multiplier
                            synth = synth_model.generate(synth_size)
                            if not isinstance(synth, pd.DataFrame):
                                synth = synth.dataframe()                            
                            synth.to_csv(synth_file, index=False)
                            print(f"    Saved {multiplier}x synthetic data ({len(synth)} rows) to: {synth_file}")
                        except Exception as e:
                            print(f"    Error generating {multiplier}x synthetic data: {e}")
                            error_file = os.path.join(seed_folder, f'generation_error_{multiplier}x.txt')
                            with open(error_file, 'w') as f:
                                f.write(f"Error generating {multiplier}x data: {str(e)}\nSeed: {model_seed}\n")

                    info_file = os.path.join(seed_folder, 'training_info.txt')
                    with open(info_file, 'w') as f:
                        f.write(f"Model: {model_name}\n")
                        f.write(f"Seed: {model_seed}\n")
                        f.write(f"Training time: {training_time:.2f} seconds\n")
                        f.write(f"Member set size: {len(mem_set)}\n")
                        f.write(f"Generated multipliers: {synth_multipliers}\n")

                    # Clean up workspace after this model run
                    workspace_dir = './workspace'
                    if os.path.exists(workspace_dir):
                        for root, dirs, files in os.walk(workspace_dir, topdown=False):
                            for name in files:
                                try:
                                    os.remove(os.path.join(root, name))
                                except Exception as e:
                                    print(f"      Failed to delete file {name} in workspace/: {e}")
                            for name in dirs:
                                try:
                                    os.rmdir(os.path.join(root, name))
                                except Exception as e:
                                    print(f"      Failed to delete directory {name} in workspace/: {e}")
                        print(f"      Cleared workspace/ after model {model_name}, seed {model_seed}")
                    diagnose_workspace('./workspace')  # See what's there first
                    cleanup_workspace_after_model(model_name, model_seed)

                    del synth_model
                    gc.collect()

                except Exception as e:
                    print(f"    Error processing model {model_name} with seed {model_seed}: {e}")
                    error_file = os.path.join(seed_folder, 'model_error.txt')
                    os.makedirs(seed_folder, exist_ok=True)
                    with open(error_file, 'w') as f:
                        f.write(f"Error processing model: {str(e)}\nSeed: {model_seed}\n")

    except Exception as e:
        print(f"Error processing dataset {data_file}: {e}")

print("\nProcessing complete!")
print("\nDirectory structure:")
print("ensemble_data")
print("├── dataset_name/")
print("│   ├── mem_set.csv")
print("│   ├── holdout_set.csv")
print("│   ├── ref_set.csv")
print("│   └── model_name/")
print("│       └── seed_X/")
print("│           ├── synth_1x.csv")
print("│           ├── synth_2x.csv")
print("│           ├── synth_3x.csv")
print("│           └── training_info.txt")
