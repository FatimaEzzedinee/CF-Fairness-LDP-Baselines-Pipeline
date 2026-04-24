# We only need level 3 (structural) counterfactuals

# Seed
import random
# random.seed(42)

# Packages
import pandas as pd
import numpy as np

import argparse

from sklearn.model_selection import train_test_split

# Setup -------------------------------------------------------------------

# working directory
import os

if __name__ == '__main__':

    # parser initialization 
    parser = argparse.ArgumentParser(description='Script pretraining bbox models')
    parser.add_argument('--dataset', type=str, default='compas', help='folktables,hospital,adult,informs,synth_adult,synth_hospital,synth_informs,compas,default_credit,synth_compas')
    parser.add_argument('--rseed', type=int, default=4, help='random seed: choose between 0 - 5')
    # parser.add_argument('--model', type=str, default='NN', help='NN, RF, SVM, XGBoost')
    # parser.add_argument('--epsilon', type=float, default='1', help='0.01, 0.1, 1, 5, 10')
    # parser.add_argument('--cf_method', type=str, default='dice_gradient', help='dice, dice_gradient, dice_kdtree, dice_genetic')
    # parser.add_argument('--seeds', type=str, default="0,1,2,3,4", help='random seed: choose between 0 - 5')

    #here we do not have k and CF  and indices since we are just training our models
    
    # get input      
    args = parser.parse_args()
    dataset = args.dataset
    seed = args.rseed
    # model = args.model
    # epsilon = args.epsilon
    # cf_method = args.cf_method
    

    # if(epsilon >= 1):
    #     epsilon = int(epsilon)
    cf_dir =  './dpnice/optimized/results/'
    path_data = './data/'
    # modeloutdir = './dpnice/optimized/pretrained/{}/'.format(dataset)
    # DSoutdir = './dpnice/optimized/datasets_loaded/{}/'.format(dataset)
    # result_file = './dpnice/optimized/results/model_accuracy.txt'
    # if not os.path.exists(modeloutdir):
        # os.makedirs(modeloutdir, exist_ok=True)

 
    # dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(dir)
    # os.chdir('..')
    # wd = os.getcwd()

    # folder paths
    # path_data = f"{wd}/data/"
    path_mdls = './src/stan_models/'
    path_rslt = './data/'

    # path_data = ds_dir
    # original data
    org_df = pd.read_csv(f"{path_data}LawSchool.csv", sep='|')

    # initial vars
    use_race = "race_nonwhite"
    vars = ["LSAT", "UGPA", "sex"]
    vars.append(use_race)
    # modified vars for counterfactual generation
    vars_m = ["LSAT", "UGPA"]

    # modeling data
    df = org_df[vars]

    # var transformation
    df["LSAT"] = df["LSAT"].round()

    sense_cols = ["female", "male"]
    print(sense_cols)
    df["female"] = (df["sex"] == "Female").astype(int)
    df["male"] = (df["sex"] == "Male").astype(int)
    print(df["sex"].value_counts())

    if use_race == "race_nonwhite":
        df["white"] = (df["race_nonwhite"] == "White").astype(int)
        sense_cols.append("white")
        df["nonwhite"] = (df["race_nonwhite"] == "NonWhite").astype(int)
        sense_cols.append("nonwhite")
        print(df["race_nonwhite"].value_counts())

    if use_race == "race_simpler":
        df["white"] = (df["race_simpler"] == "White").astype(int)
        sense_cols.append("white")
        df["black"] = (df["race_simpler"] == "Black").astype(int)
        sense_cols.append("black")
        df["Latino"] = (df["race_simpler"] == "Latino").astype(int)
        sense_cols.append("latino")
        df["asian"] = (df["race_simpler"] == "Asian").astype(int)
        sense_cols.append("asian")
        df["other"] = (df["race_simpler"] == "Other").astype(int)
        sense_cols.append("Other")
        print(df["race_simpler"].value_counts())

    vars_m.extend(sense_cols)

    # Level 3 -----------------------------------------------------------------

    # DAG: Sex -> UGPA; Race -> UGPA; Sex -> LSAT; Race -> LSAT
    df_lev3 = df.copy()

    # Step 1: train model for descendant nodes
    import statsmodels.api as sm

    model_ugpa = sm.OLS(df_lev3["UGPA"], sm.add_constant(df_lev3[["female", "nonwhite"]])).fit()
    print(model_ugpa.summary())

    model_lsat = sm.OLS(df_lev3["LSAT"], sm.add_constant(df_lev3[["female", "nonwhite"]])).fit()
    print(model_lsat.summary())

    # perform the abduction step: estimate the residuals
    df_lev3["resid_UGPA"] = df_lev3["UGPA"] - model_ugpa.predict(sm.add_constant(df_lev3[["female", "nonwhite"]]))
    df_lev3["resid_LSAT"] = df_lev3["LSAT"] - model_lsat.predict(sm.add_constant(df_lev3[["female", "nonwhite"]]))

    # Step 2: action on race and gender (accordingly: under multiple disc.)
    # do(Gender:='Male')
    df_lev3_do_male = pd.DataFrame({
        "female": np.zeros(len(df_lev3)),
        "nonwhite": df_lev3["nonwhite"]
    })

    # do(Race:='White')
    df_lev3_do_white = pd.DataFrame({
        "female": df_lev3["female"],
        "nonwhite": np.zeros(len(df_lev3))
    })

    # Step 3: prediction
    # do(Gender:='Male')
    df_lev3_do_male["Sex"] = df_lev3["sex"]
    df_lev3_do_male["Race"] = df_lev3["race_nonwhite"]
    df_lev3_do_male["resid_LSAT"] = df_lev3["resid_LSAT"]
    df_lev3_do_male["resid_UGPA"] = df_lev3["resid_UGPA"]

    df_lev3_do_male["scf_LSAT"] = (model_lsat.predict(sm.add_constant(df_lev3_do_male[["female", "nonwhite"]])) + df_lev3_do_male["resid_LSAT"]).round(3)
    df_lev3_do_male["scf_UGPA"] = (model_ugpa.predict(sm.add_constant(df_lev3_do_male[["female", "nonwhite"]])) + df_lev3_do_male["resid_UGPA"]).round(3)

    print(df_lev3_do_male["scf_LSAT"].describe())
    print(df_lev3_do_male["scf_UGPA"].describe())

    df_lev3_do_male["scf_LSAT"] = np.clip(df_lev3_do_male["scf_LSAT"], 10.00, 48.00)
    df_lev3_do_male["scf_UGPA"] = np.clip(df_lev3_do_male["scf_UGPA"], 0.00, 4.00)

    print(df_lev3_do_male["scf_LSAT"].describe())
    print(df_lev3_do_male["scf_UGPA"].describe())

    df_lev3_do_male.to_csv(f"{path_rslt}cf_LawSchool_lev3_doMale.csv", sep="|", index=False)

    # do(Race:='White')
    df_lev3_do_white["Sex"] = df_lev3["sex"]
    df_lev3_do_white["Race"] = df_lev3["race_nonwhite"]
    df_lev3_do_white["resid_LSAT"] = df_lev3["resid_LSAT"]
    df_lev3_do_white["resid_UGPA"] = df_lev3["resid_UGPA"]

    df_lev3_do_white["scf_LSAT"] = (model_lsat.predict(sm.add_constant(df_lev3_do_white[["female", "nonwhite"]])) + df_lev3_do_white["resid_LSAT"]).round(3)
    df_lev3_do_white["scf_UGPA"] = (model_ugpa.predict(sm.add_constant(df_lev3_do_white[["female", "nonwhite"]])) + df_lev3_do_white["resid_UGPA"]).round(3)

    print(df_lev3_do_white["scf_LSAT"].describe())
    print(df_lev3_do_white["scf_UGPA"].describe())

    df_lev3_do_white["scf_LSAT"] = np.clip(df_lev3_do_white["scf_LSAT"], 10.00, 48.00)
    df_lev3_do_white["scf_UGPA"] = np.clip(df_lev3_do_white["scf_UGPA"], 0.00, 4.00)

    print(df_lev3_do_white["scf_LSAT"].describe())
    print(df_lev3_do_white["scf_UGPA"].describe())

    df_lev3_do_white.to_csv(f"{path_rslt}cf_LawSchool_lev3_doWhite.csv", sep="|", index=False)

# New function to debias the model
def debias_model():
    """
    This function debiases the model by adding counterfactual predictions to the dataset,
    retraining the model, and saving the debiased dataset.
    """
    # Add counterfactual predictions to the original dataset
    df["cf_LSAT_male"] = df_lev3_do_male["scf_LSAT"]
    df["cf_UGPA_male"] = df_lev3_do_male["scf_UGPA"]

    df["cf_LSAT_white"] = df_lev3_do_white["scf_LSAT"]
    df["cf_UGPA_white"] = df_lev3_do_white["scf_UGPA"]

    # Combine original and counterfactual data
    new_data = pd.concat([df, df_lev3_do_male, df_lev3_do_white], axis=0)

    # Train the model on the new dataset
    model_ugpa_debiased = sm.OLS(new_data["UGPA"], sm.add_constant(new_data[["female", "nonwhite"]])).fit()
    model_lsat_debiased = sm.OLS(new_data["LSAT"], sm.add_constant(new_data[["female", "nonwhite"]])).fit()

    # Print summaries of the debiased models
    print("Debiased UGPA Model Summary:")
    print(model_ugpa_debiased.summary())

    print("Debiased LSAT Model Summary:")
    print(model_lsat_debiased.summary())

    # Save the new dataset with counterfactual predictions
    new_data.to_csv(f"{path_rslt}debiased_LawSchool_data.csv", sep="|", index=False)

# Call the new function to debias the model
debias_model()