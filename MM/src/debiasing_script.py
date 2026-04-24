import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import argparse

from utils.data.load_data import Fetcher
def train_model_without_bias(dataset, target_column):
    """
    Train a model without using sex, race, or id columns.
    """
    features = dataset.drop(columns=["sex", "race", "id", target_column])
    target = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, predictions)}")

    return model, X_test, y_test

def debias_dataset(test_set, comparator_file, sensitive_column, sensitive_value, new_file_name, model, target_column):
    """
    Debias the dataset based on a sensitive attribute (e.g., gender or race).
    """
    comparator_data = pd.read_csv(comparator_file)
    updated_data = [] # test_set.copy()
    columns_to_remove = ["sex", "race", "id"]
    for index, row in test_set.iterrows():
        if row[sensitive_column] != sensitive_value:
            to_predict_row = row.drop(labels=columns_to_remove )
            # .drop(labels=["sex", "race", "id"])
            # Ensure the input retains feature names to avoid warnings
            decision = model.predict(to_predict_row.to_frame().T)[0]
            if decision == 0:  # Negative decision
                comparator_row = comparator_data[comparator_data["id"] == row["id"]]
                
                if not comparator_row.empty:
                    
                    pure_comparator_row = comparator_row.drop(columns=columns_to_remove)
                    combined_df = pd.concat([row.to_frame().T, comparator_row], ignore_index=True)
                    print("Combined DataFrame for row and comparator_row:")
                    print(combined_df)
                    comparator_decision = model.predict(pure_comparator_row)[0]
                    
                    if comparator_decision != decision:  # Decision changes
                        new_row = row.copy()
                        new_row[target_column] = 1  # Update label to positive
                        # updated_data = pd.concat([updated_data, new_row.to_frame().T], ignore_index=True)
                        updated_data.append(new_row)
    updated_data = pd.DataFrame(updated_data)
                        # Combine row and comparator_row into a single DataFrame for comparison
                        

    # Save the updated dataset
    updated_data.to_csv(new_file_name, index=False)
    print(f"Saved debiased dataset to {new_file_name}")

    # Train a new model on the updated dataset
    # new_model, _, _ = train_model_without_bias(updated_data, target_column)
    return #new_model

# Function to train a model with a 60-40 train-test split
def train_model_with_split(dataset, target,rseed):
    """
    Train a model with a 60-40 train-test split, excluding race, sex, and id.
    """
    # features = dataset.drop(columns=["sex", "race", "id"])
    # target = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.4, random_state=rseed)
    X_train_features = X_train.drop(columns=["sex", "race", "id"])
    X_test_features = X_test.drop(columns=["sex", "race", "id"])   
    model = LogisticRegression()
    model.fit(X_train_features, y_train)

    # Evaluate the model
    predictions = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained with 60-40 split. Accuracy: {accuracy}")

    return model, X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='adult', help='folktables,hospital,adult,informs,synth_adult,synth_hospital,synth_informs,compas,default_credit,synth_compas')
    parser.add_argument('--rseed', type=int, default=0, help='random seed: choose between 0 - 5')

    args = parser.parse_args()
    dataset_name = args.dataset
    rseed = args.rseed

    path_results = "./results/"
    datasets_info = {
        "adult": ["target"],# ["workclass", "education", "occupation", "hours-per-week"],
    }

    
    # for dependent_columns in datasets_info.items():
        # for target_column in dependent_columns:
    print(f"Processing debiasing for dataset: {dataset_name}")

    # Load dataset
    fetcher = Fetcher(name=dataset_name, use_id =True)
    dataset = fetcher.dataset['X']  # Assuming Fetcher provides the dataset as an attribute
    label = fetcher.dataset['y']  # Assuming Fetcher provides labels as well

    # dataset = pd.read_csv(f"./data/{dataset_name}.csv")
    # target_column = datasets_info[dataset_name][0]  # Assuming one target column per dataset
    # dataset = dataset.dropna()  # Drop rows with missing values for simplicity
    # Train initial model
    model, _, X_test, _ ,  y_test =  train_model_with_split(dataset, label, rseed) #  train_model_without_bias(dataset, target_column)

    # Gender debiasing
    debias_dataset(
        X_test,
        f"{path_results}male_comparators_{dataset_name}.csv",
        "sex",
        1,    # male value is 1 in the dataset
        f"{path_results}gender_debiased_{dataset_name}.csv",
        model,
        label
    )

    # gender_debiased_model = debias_dataset(
    #     X_test,
    #     f"{path_results}male_comparators_{dataset_name}.csv",
    #     "sex",
    #     1,    # male value is 1 in the dataset
    #     f"{path_results}gender_debiased_{dataset_name}.csv",
    #     model,
    #     label
    # )

    # Race debiasing
    debias_dataset(
        X_test,
        f"{path_results}white_comparators_{dataset_name}.csv",
        "race",
        4,   # white value is 4 in the dataset
        f"{path_results}race_debiased_{dataset_name}.csv",
        model,
        label
    )

    # race_debiased_model = debias_dataset(
    #     X_test,
    #     f"{path_results}white_comparators_{dataset_name}.csv",
    #     "race",
    #     4,   # white value is 4 in the dataset
    #     f"{path_results}race_debiased_{dataset_name}.csv",
    #     model,
    #     label
    # )

    # Train a model with a 60-40 split
    # model, X_train, X_test, y_train, y_test = train_model_with_split(dataset, label)