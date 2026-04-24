"""
Data preprocessing and file loading utilities for MIA evaluation
"""
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def fit_transformer(X,numerical_features=None):
    """
    Fits a transformer on the reference dataset and stores column information.
    Ensures categorical variables are both encoded and scaled.
   
    Parameters:
    X (pd.DataFrame): Reference dataset to fit the transformer
   
    Returns:
    tuple: (fitted_transformer, feature_names)
    """
    # Check for None input
    if X is None:
        raise ValueError("Input DataFrame X is None. Please check your data loading step.")
    # Separate the numerical and categorical columns
    # A column is numerical if it has more than 10 unique values, otherwise categorical
    numerical_cols = X.columns[X.nunique() > 10]
    categorical_cols = X.columns[X.nunique() <= 10]
   
    # A column is numerical if it has more than 10 unique values, otherwise categorical
    # numerical_cols = X.columns[X.nunique() > 10]
    # categorical_cols = X.columns[X.nunique() <= 10]

    # original code to find categorical and numerical columns
    # numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    # categorical_cols = X.select_dtypes(include=['object']).columns
    # Create transformers for numerical and categorical data
    numerical_transformer = StandardScaler()
    
    # For categorical data: first encode, then scale
    categorical_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scaler', StandardScaler())  # Add scaling after encoding
    ])
   
    # Create the full column transformer
    full_transformer = ColumnTransformer([
        ('numerical', numerical_transformer, numerical_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ], remainder='drop')
   
    # Fit the transformer and get feature names
    full_transformer.fit(X)
   
    # Get feature names for numerical and categorical data
    numeric_features = numerical_cols.tolist()
    categorical_features = categorical_cols.tolist()
   
    # Combine feature names
    feature_names = numeric_features + categorical_features
   
    return full_transformer, feature_names


def transform_data(X, transformer_info):
    """
    Transforms data using a pre-fitted transformer and ensures consistent columns.
    
    Parameters:
    X (pd.DataFrame): Dataset to transform
    transformer_info: tuple of (transformer, feature_names)
    
    Returns:
    np.ndarray: Transformed data with consistent columns
    """
    transformer, feature_names = transformer_info
    transformed_data = transformer.transform(X)
    
    # Ensure the transformed data has the correct number of columns
    if transformed_data.shape[1] != len(feature_names):
        # If we have fewer columns than expected, pad with zeros
        padding = np.zeros((transformed_data.shape[0], len(feature_names) - transformed_data.shape[1]))
        transformed_data = np.hstack([transformed_data, padding])
    
    return transformed_data


def ordinal_tabular_preprocess(mem, non_mem, synth, ref, fit_target='synth',numerical_features=None):
    """
    Preprocesses tabular data for MIA evaluation.
    
    Parameters:
    mem, non_mem, synth, ref: DataFrames for member, non-member, synthetic, and reference data
    fit_target: str, either 'synth' or 'ref' to prevent data leakage
    
    Returns:
    tuple: (transformed_mem, transformed_non_mem, transformed_synth, transformed_ref, transformer)
    """
    if fit_target not in ['synth', 'ref']:
        raise ValueError("fit_target must be 'synth' or 'ref' to prevent data leakage.")
    
    fit_data = synth if fit_target == 'synth' else ref
    transformer = fit_transformer(fit_data,numerical_features)
    
    datasets = [mem, non_mem, synth, ref]
    transformed = [transform_data(data, transformer) for data in datasets]
    
    return (*transformed, transformer)


def load_csv_file(directory, base_pattern, label):
    """
    Loads a CSV file that starts with the specified base pattern, ignoring any numbers that follow.
    
    Args:
        directory (Path): Directory to search in
        base_pattern (str): Base pattern to look for at start of filename (e.g., 'mem_set')
        label (str): Label for the loaded dataset
    
    Returns:
        DataFrame or None: Loaded data if file is found, None otherwise
    """
    if directory.exists():
        for file in directory.glob(f'{base_pattern}.csv'):
            print(f"Loaded: {file.name} as '{label}'")
            return pd.read_csv(file)
        for file in directory.glob(f'{base_pattern}_*.csv'):
            print(f"Loaded: {file.name} as '{label}'")
            return pd.read_csv(file)
    return None


def load_required_files(base_path, dataset, seed, method,model, size , synth_size=1500):
    """
    Searches for and loads required CSV files into a dictionary of DataFrames.
    Modified to work without n_size directory level.

    Args:
        base_path (Path): The root directory path to start searching from.
        dataset (str): The dataset name.
        seed (str): The seed folder name.
        method (str): The method folder name (e.g., 'ddpm').

    Returns:
        dict: A dictionary containing loaded DataFrames.
    """
    dfs = {}
    dataset_path = base_path / dataset /model

    # Search for required files in specified seed folder (without n_size)
    # seed_path = dataset_path / seed / method
    seed_path = dataset_path / str(seed) 
    if seed_path.exists():
        # Load files with specific base patterns, ignoring trailing numbers
        dfs['mem'] = load_csv_file(seed_path, 'mem_set', 'mem')
        if len(dfs['mem']) > size:
                    dfs['mem'] = dfs['mem'].iloc[:size]
        holdout_set = load_csv_file(seed_path, 'holdout_set', 'holdout_set')
        
        if holdout_set is not None:
            holdout_set_shuffled = holdout_set.sample(frac=1, random_state=42).reset_index(drop=True)
            
            #### added by maryam
            if len(holdout_set_shuffled) >= 2 * size:
                holdout_set_shuffled = holdout_set_shuffled.iloc[:2 * size] 
            else:
                raise ValueError(f"Holdout set has insufficient rows: {len(holdout_set_shuffled)} rows found, but at least {2 * size} required.")
            ##### end added by maryam
            # Calculate split point
            split_point = len(holdout_set_shuffled) // 2
            

            # Split into two parts
            dfs['non_mem'] = holdout_set_shuffled.iloc[:split_point]
            dfs['ref'] = holdout_set_shuffled.iloc[split_point:split_point + size]

        # Look for synthetic data in method-specific folder
        # synth_path = seed_path / 'synth' / method
        synth_path = seed_path / method  # Adjusted to look directly in method folder
        if synth_path.exists():
            dfs['synth'] = load_csv_file(synth_path, 'synth_1x', 'synth')
            if dfs['synth'] is not None and 'is_member' in dfs['synth'].columns:
                dfs['synth'] = dfs['synth'].drop(columns=['is_member'])
            if len(dfs['synth']) > synth_size:
                    dfs['synth'] = dfs['synth'].iloc[:synth_size]
    # Verify all required files were loaded
    required_keys = {'mem', 'non_mem', 'synth', 'ref'}
    missing_keys = required_keys - set(dfs.keys())
    if missing_keys:
        raise FileNotFoundError(f"Missing required files: {missing_keys}")

    return dfs
