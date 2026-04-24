from ast import main

import pandas as pd
import numpy as np
import os
from utils.data.load_data import Fetcher
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import argparse

# augmentation_method = "add_comparators"  # Options: "update_labels", "add_comparators", "combine"
# Define datasets and their dependent columns
datasets_info = {
    "adult": ["workclass", "education", "occupation", "hours-per-week"],
    "compas": ["c_charge_degree_enc","priors_count","juv_fel_count","juv_misd_count", "juv_other_count"]
    # Add more datasets and their dependent columns here
}
sensitive_columns = {
    "adult": ["sex", "race"],
    "compas": ["race_enc"] # "sex_enc",
}

# Define values for Male and White for each dataset
male_values = {
    "adult": 1.0,
    "compas": 1.0
}
female_values = {
    "adult": 0.0,
    "compas": 0.0
}
white_values = {
    "adult": 4.0,
    "compas": 1.0
}


# Process each dataset
def main_MM(augmentation_method, dataset_name):
    # Paths
    path_data = "./data/"
    path_results = "./results/"

    # Ensure results directory exists
    os.makedirs(path_results, exist_ok=True)

    # Initialize lists to store data for white and male comparators
    white_comparator_data = []
    male_comparator_data = []

    dependent_columns = datasets_info.get(dataset_name, [])
    sen_columns = sensitive_columns.get(dataset_name, [])
    # for dependent_columns in dependent_columns.items():
    print(f"Processing dataset: {dataset_name}")

    # Fetch dataset
    fetcher = Fetcher(name=dataset_name)
    dataset = fetcher.dataset['X']  # Assuming Fetcher provides the dataset as an attribute
    label = fetcher.dataset['y']  # Assuming Fetcher provides labels as well
    # Add an ID column
    dataset['id'] = range(1, len(dataset) + 1)
    
    dataset['target'] = label  # Add target variable to the dataset
    dataset.to_csv(f"{path_data}{dataset_name}_with_ids.csv", index=False)
    dataset = dataset.drop(columns=["target"])  # Drop target variable for modeling
    dataset.reset_index(drop=True, inplace=True)  # Reset index after adding ID column

    white_comparator_data = dataset.copy()  # Initialize with original dataset for white comparators
    male_comparator_data = dataset.copy()  # Initialize with original dataset for male comparators

    # Process each dependent column
    # dep columns are the cf columns we can do changes to
    for dep_col in dependent_columns:
            print(f"Processing dependent column: {dep_col}")

            # Ensure required columns exist
            
            if not all(col in dataset.columns for col in sen_columns + ["age", dep_col]):
            # if not all(col in dataset.columns for col in [sen_columns, "age", dep_col]):
                print(f"Skipping {dep_col} as required columns are missing.")
                continue

            # Check if 'age' exists in the dataset columns
            if 'age' in dataset.columns:
                # Prepare data with age
                df = dataset[[sensitive_columns[dataset_name][0], sensitive_columns[dataset_name][1], "age", dep_col, "id"]].copy()
                df["female"] = (df[sensitive_columns[dataset_name][0]] == female_values[dataset_name]).astype(int)
                df["male"] = (df[sensitive_columns[dataset_name][0]] == male_values[dataset_name]).astype(int)
                df["nonwhite"] = (df[sensitive_columns[dataset_name][1]] != white_values[dataset_name]).astype(int)

                # Train regression model with age
                model = sm.OLS(df[dep_col], sm.add_constant(df[["female", "nonwhite", "age"]])).fit()
                print(model.summary())

                # Abduction step: estimate residuals
                df[f"resid_{dep_col}"] = df[dep_col] - model.predict(sm.add_constant(df[["female", "nonwhite", "age"]]))

                # Generate counterfactuals
                for action, action_values in {"do_male": {"female": 0}, "do_white": {"nonwhite": 0}}.items():
                    cf_df = df.copy()
                    for col, value in action_values.items():
                        cf_df[col] = value
                    train_columns = ["female", "nonwhite", "age"]

                    cf_df[f"scf_{dep_col}"] = model.predict(sm.add_constant(cf_df[train_columns])) + cf_df[f"resid_{dep_col}"]
                    # Round the predicted values to integers
                    cf_df[f"scf_{dep_col}"] = cf_df[f"scf_{dep_col}"].round().astype(int)

                    # Save results
                    output_file = f"{path_results}{action}_{dataset_name}_{dep_col}.csv"
                    cf_df.to_csv(output_file, index=False)
                    print(f"Saved {action} results to {output_file}")
            else:
                # Prepare data without age
                df = dataset[[sensitive_columns[dataset_name][0], sensitive_columns[dataset_name][1], dep_col, "id"]].copy()
                df["female"] = (df[sensitive_columns[dataset_name][0]] == female_values[dataset_name]).astype(int)
                df["male"] = (df[sensitive_columns[dataset_name][0]] == male_values[dataset_name]).astype(int)
                df["nonwhite"] = (df[sensitive_columns[dataset_name][1]] != white_values[dataset_name]).astype(int)

                # Train regression model without age
                model = sm.OLS(df[dep_col], sm.add_constant(df[["female", "nonwhite"]])).fit()
                print(model.summary())

                # Abduction step: estimate residuals
                df[f"resid_{dep_col}"] = df[dep_col] - model.predict(sm.add_constant(df[["female", "nonwhite"]]))

                # Generate counterfactuals
                for action, action_values in {"do_male": {"female": 0}, "do_white": {"nonwhite": 0}}.items():
                    cf_df = df.copy()
                    for col, value in action_values.items():
                        cf_df[col] = value
                    train_columns = ["female", "nonwhite"]

                    cf_df[f"scf_{dep_col}"] = model.predict(sm.add_constant(cf_df[train_columns])) + cf_df[f"resid_{dep_col}"]
                    # Round the predicted values to integers
                    cf_df[f"scf_{dep_col}"] = cf_df[f"scf_{dep_col}"].round().astype(int)

                    # Save results
                    output_file = f"{path_results}{action}_{dataset_name}_{dep_col}.csv"
                    cf_df.to_csv(output_file, index=False)
                    print(f"Saved {action} results to {output_file}")

        # After processing all dependent columns, merge and save comparator files
        # if white_comparator_data:
            # white_merged = pd.concat(white_comparator_data).drop_duplicates(subset="id")
        
        # Build white comparator data from do_white results
    white_comparator_data = dataset.copy()
    augmented_set = pd.DataFrame()  # Initialize an empty DataFrame to store augmented data
    for dep_col in dependent_columns:
        do_white_file = f"{path_results}do_white_{dataset_name}_{dep_col}.csv"
        if os.path.exists(do_white_file):
            do_white_df = pd.read_csv(do_white_file)
            # Merge the scf values into white_comparator_data
            white_comparator_data = white_comparator_data.merge(
                do_white_df[["id", f"scf_{dep_col}"]],
                on="id",
                how="left"
            )
            # Update the dependent column with the scf value where race != white
            
            white_comparator_data.loc[:, dep_col] = white_comparator_data.loc[:, f"scf_{dep_col}"]

            # Drop the temporary scf column
            white_comparator_data = white_comparator_data.drop(columns=[f"scf_{dep_col}"])
    white_comparator_data.to_csv(f"{path_results}white_comparators_{dataset_name}.csv", index=False)
    print(f"Saved white comparators to {path_results}white_comparators_{dataset_name}.csv")
            ## Decide on the augmentation method:
            ### 1. add query rows to the original dataset whit comparator lables
            ### 2. Add comparator with the oridinal label
            ### 3. 
            ###### Create augmented set by matching rows in white_comparator_data with original dataset based on common features (excluding id and target)
    
    if augmentation_method == "update_labels":
        # Build original data with IDs + targets
        original_df = fetcher.dataset["X"].copy()
        original_df["id"] = range(1, len(original_df) + 1)
        original_df["target"] = label.values if hasattr(label, "values") else label
        common_cols = [c for c in white_comparator_data.columns if c in original_df.columns and c not in {"id", "target"}]
        # Start with original targets by id (used for white rows)
        id_to_target = original_df.set_index("id")["target"]
        desired_targets = white_comparator_data["id"].map(id_to_target)

        # White rows keep their original target
        white_mask = white_comparator_data[sensitive_columns[dataset_name][1]].astype(float) == float(white_values[dataset_name])

        # Non-white rows: assign label from nearest white neighbor in original set
        original_whites = original_df[original_df[sensitive_columns[dataset_name][1]].astype(float) == float(white_values[dataset_name])].copy()
        nonwhite_idx = white_comparator_data.index[~white_mask]

        if len(nonwhite_idx) > 0 and not original_whites.empty:
            X_white = pd.get_dummies(original_whites[common_cols], drop_first=False)
            X_nonwhite = pd.get_dummies(
                white_comparator_data.loc[nonwhite_idx, common_cols], drop_first=False
            ).reindex(columns=X_white.columns, fill_value=0)

            nn_features = [c for c in common_cols if c not in {sensitive_columns[dataset_name][0], sensitive_columns[dataset_name][1]}]

            X_white = pd.get_dummies(original_whites[nn_features], drop_first=False)
            X_nonwhite = pd.get_dummies(
                white_comparator_data.loc[nonwhite_idx, nn_features], drop_first=False
            ).reindex(columns=X_white.columns, fill_value=0)

            nn_white = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn_white.fit(X_white)
            _, idx_white = nn_white.kneighbors(X_nonwhite)

            desired_targets.loc[nonwhite_idx] = (
                original_whites.iloc[idx_white.flatten()]["target"].to_numpy()
            )

        # Force downstream NN block to copy `desired_targets` exactly
        original_df["target"] = desired_targets.reset_index(drop=True).copy()
        
        augmented_set = original_df.copy()# pd.concat([augmented_set, white_comparator_data], ignore_index=True)
        print("number of positive labels in augmented set:", augmented_set["target"].sum())
        print("number of positive labels in original set:", label.sum())
        augmented_set.to_csv(f"{path_results}augmented_set_{dataset_name}_{augmentation_method}.csv", index=False)
    
    elif augmentation_method == "add_comparators":
        # Add comparator rows to the original dataset
        original_df = fetcher.dataset["X"].copy()
        # original_df["id"] = range(1, len(original_df) + 1)
        original_df["id_target"] = label.values if hasattr(label, "values") else label

        # For non-white rows in white_comparator_data, add comparator rows with white race
        nonwhite_mask = white_comparator_data[sensitive_columns[dataset_name][1]].astype(float) != float(white_values[dataset_name])
        nonwhite_rows = white_comparator_data[nonwhite_mask].copy()

        # Create comparator rows: change race to white and use original target
        id_to_target = original_df.set_index("id")["id_target"]
        nonwhite_rows[sensitive_columns[dataset_name][1]] = white_values[dataset_name]
        nonwhite_rows["target"] = nonwhite_rows["id"].map(id_to_target)
        
        
        # Drop extra columns from nonwhite_rows and align to original_df
        nonwhite_rows = nonwhite_rows.reindex(columns=original_df.columns, fill_value=np.nan)

        augmented_set = pd.concat([original_df, nonwhite_rows], ignore_index=True)
        print("number of positive labels in augmented set:", augmented_set["target"].sum())
        print("number of positive labels in original set:", label.sum())
        augmented_set.to_csv(f"{path_results}augmented_set_white_{dataset_name}_{augmentation_method}.csv", index=False)

    # Build male comparator data from do_male outputs (for augmentation, analogous to white flow)
    #######################################################
    # if male_comparator_data:
        # male_merged = pd.concat(male_comparator_data).drop_duplicates(subset="id")
    male_comparator_data = dataset.copy()
    
    for dep_col in dependent_columns:
        do_male_file = f"{path_results}do_male_{dataset_name}_{dep_col}.csv"
        if os.path.exists(do_male_file):
            do_male_df = pd.read_csv(do_male_file)
            # Merge the scf values into male_comparator_data
            male_comparator_data = male_comparator_data.merge(
                do_male_df[["id", f"scf_{dep_col}"]],
                on="id",
                how="left"
            )
            # Update the dependent column with the scf value where sex != male
            # mask = male_comparator_data["sex"] != male_values[dataset_name]
            male_comparator_data.loc[:, dep_col] = male_comparator_data.loc[:, f"scf_{dep_col}"]
            male_comparator_data.loc[:, sensitive_columns[dataset_name][0]] = male_values[dataset_name]
            # Drop the temporary scf column
            male_comparator_data = male_comparator_data.drop(columns=[f"scf_{dep_col}"])
    male_comparator_data.to_csv(f"{path_results}male_comparators_{dataset_name}.csv", index=False)
    print(f"Saved male comparators to {path_results}male_comparators_{dataset_name}.csv")

    if augmentation_method == "update_labels":
        original_df_m = fetcher.dataset["X"].copy()
        original_df_m["id"] = range(1, len(original_df_m) + 1)
        original_df_m["target"] = label.values if hasattr(label, "values") else label

        common_cols_m = [
            c for c in male_comparator_data.columns
            if c in original_df_m.columns and c not in {"id", "target"}
        ]

        id_to_target_m = original_df_m.set_index("id")["target"]
        desired_targets_m = male_comparator_data["id"].map(id_to_target_m)

        male_mask = original_df_m[sensitive_columns[dataset_name][0]].astype(float) == float(male_values[dataset_name])
        nonmale_idx = original_df_m.index[~male_mask]

        original_males = original_df_m[male_mask].copy()
        if len(nonmale_idx) > 0 and not original_males.empty:
            nn_features_m = [c for c in common_cols_m if c not in {sensitive_columns[dataset_name][0], sensitive_columns[dataset_name][1]}]

            X_male = pd.get_dummies(original_males[nn_features_m], drop_first=False)
            X_nonmale = pd.get_dummies(
                male_comparator_data.loc[nonmale_idx, nn_features_m], drop_first=False
            ).reindex(columns=X_male.columns, fill_value=0)

            nn_male = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn_male.fit(X_male)
            _, idx_male = nn_male.kneighbors(X_nonmale)

            desired_targets_m.loc[nonmale_idx] = (
                original_males.iloc[idx_male.flatten()]["target"].to_numpy()
            )

        male_augmented_set = original_df_m.copy()
        male_augmented_set["target"] = desired_targets_m.to_numpy()
        print("number of positive labels in augmented set:", male_augmented_set["target"].sum())
        print("number of positive labels in original set:", label.sum())

        # final concatenates set (initial + MM) -> Fatima: take this file as input
        male_augmented_set.to_csv(
            f"{path_results}augmented_set_male_{dataset_name}_{augmentation_method}.csv",
            index=False,
        )

    elif augmentation_method == "add_comparators":
        original_df_m = fetcher.dataset["X"].copy()
        original_df_m["id"] = range(1, len(original_df_m) + 1)
        original_df_m["target"] = label.values if hasattr(label, "values") else label

        nonmale_ids = original_df_m.loc[
            original_df_m[sensitive_columns[dataset_name][0]].astype(float) != float(male_values[dataset_name]), "id"
        ]
        nonmale_rows = male_comparator_data[male_comparator_data["id"].isin(nonmale_ids)].copy()
        nonmale_rows[sensitive_columns[dataset_name][0]] = male_values[dataset_name]
        nonmale_rows["target"] = nonmale_rows["id"].map(
            original_df_m.set_index("id")["target"]
        )
        nonmale_rows = nonmale_rows.reindex(columns=original_df_m.columns, fill_value=np.nan)

        male_augmented_set = pd.concat([original_df_m, nonmale_rows], ignore_index=True)
        print("number of positive labels in augmented set:", male_augmented_set["target"].sum())
        print("number of positive labels in original set:", label.sum())

        # final concatenates set (initial + MM) -> Fatima: take this file as input
        male_augmented_set.to_csv(
            f"{path_results}augmented_set_male_{dataset_name}_{augmentation_method}.csv",
            index=False,
        )
    


if __name__ == "__main__":
    # parser initialization 
    parser = argparse.ArgumentParser(description='Script for generating counterfactual data and debiasing')
    parser.add_argument('--augmentation_method', type=str, default='update_labels', help='Method for augmentation: update_labels, add_comparators')
    parser.add_argument('--dataset_name', type=str, default='compas', help='Name of the dataset to process')
    args = parser.parse_args()
    augmentation_method = args.augmentation_method
    dataset_name = args.dataset_name
    
    main_MM(augmentation_method, dataset_name)
