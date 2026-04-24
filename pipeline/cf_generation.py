"""
cf_generation.py -- Level-3 structural (mutatis-mutandis) counterfactual generation
for the COMPAS dataset.  Implements Pearl Abduction-Action-Prediction.

Causal DAG (race-based): Race -> priors_count, juv_fel_count, juv_misd_count, juv_other_count
Interventions: doWhite (Black->White) for Black rows, doBlack (White->Black) for White rows
Variables NOT affected: age, c_charge_degree_enc, sex_enc
"""

from __future__ import annotations
import os, sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import FEATURE_COLS, RACE_ENC_COL, PROTECTED, SEX_ENC_COL, datasets_info, TARGET_COL, sensitive_columns, female_values, male_values, white_values, DATA_DIR, OUTPUT_DIR, AUG_RELABEL_DISTANCE_PERCENTILE, AUG_RELABEL_K_NEIGHBORS, AUG_RELABEL_AGREEMENT_THRESHOLD, AUG_COMPARATORS_BIDIRECTIONAL

_PRIORS_IDX    = FEATURE_COLS.index("priors_count")
_JUV_FEL_IDX  = FEATURE_COLS.index("juv_fel_count")
_JUV_MISD_IDX = FEATURE_COLS.index("juv_misd_count")
_JUV_OTHER_IDX= FEATURE_COLS.index("juv_other_count")
_RACE_IDX      = FEATURE_COLS.index(RACE_ENC_COL)


@dataclass
class SCMParams:
    coef_priors:    np.ndarray
    coef_juv_fel:   np.ndarray
    coef_juv_misd:  np.ndarray
    coef_juv_other: np.ndarray
    resid_priors:    np.ndarray
    resid_juv_fel:   np.ndarray
    resid_juv_misd:  np.ndarray
    resid_juv_other: np.ndarray
    max_priors:    float
    max_juv_fel:   float
    max_juv_misd:  float
    max_juv_other: float
    n_train: int


def fit_scm(X_train: np.ndarray, verbose: bool = False) -> SCMParams:
    X = np.asarray(X_train, dtype=float)
    N = len(X)
    black  = X[:, _RACE_IDX]
    priors = X[:, _PRIORS_IDX]
    juv_f  = X[:, _JUV_FEL_IDX]
    juv_m  = X[:, _JUV_MISD_IDX]
    juv_o  = X[:, _JUV_OTHER_IDX]
    A = np.column_stack([np.ones(N), black])
    coef_priors, _, _, _ = np.linalg.lstsq(A, priors, rcond=None)
    coef_juv_f,  _, _, _ = np.linalg.lstsq(A, juv_f,  rcond=None)
    coef_juv_m,  _, _, _ = np.linalg.lstsq(A, juv_m,  rcond=None)
    coef_juv_o,  _, _, _ = np.linalg.lstsq(A, juv_o,  rcond=None)
    resid_priors = priors - (A @ coef_priors)
    resid_juv_f  = juv_f  - (A @ coef_juv_f)
    resid_juv_m  = juv_m  - (A @ coef_juv_m)
    resid_juv_o  = juv_o  - (A @ coef_juv_o)
    if verbose:
        b0p, bbp = coef_priors
        b0f, bbf = coef_juv_f
        b0m, bbm = coef_juv_m
        b0o, bbo = coef_juv_o
        print(f"[SCM] priors_count    = {b0p:.4f} + {bbp:.4f}*black")
        print(f"[SCM] juv_fel_count   = {b0f:.4f} + {bbf:.4f}*black")
        print(f"[SCM] juv_misd_count  = {b0m:.4f} + {bbm:.4f}*black")
        print(f"[SCM] juv_other_count = {b0o:.4f} + {bbo:.4f}*black")
        print(f"[SCM] priors residual range: [{resid_priors.min():.3f}, {resid_priors.max():.3f}]")
    return SCMParams(
        coef_priors=coef_priors, coef_juv_fel=coef_juv_f,
        coef_juv_misd=coef_juv_m, coef_juv_other=coef_juv_o,
        resid_priors=resid_priors, resid_juv_fel=resid_juv_f,
        resid_juv_misd=resid_juv_m, resid_juv_other=resid_juv_o,
        max_priors=float(priors.max()), max_juv_fel=float(juv_f.max()),
        max_juv_misd=float(juv_m.max()), max_juv_other=float(juv_o.max()),
        n_train=N,
    )


def _apply_intervention(X, rp, rf, rm, ro, cp, cf, cm, co, black_value, mp, mf, mm, mo):
    N = len(X)
    A_cf = np.column_stack([np.ones(N), np.full(N, black_value)])
    X_new = X.copy()
    X_new[:, _PRIORS_IDX]    = np.clip(np.round(A_cf @ cp + rp), 0.0, mp)
    X_new[:, _JUV_FEL_IDX]  = np.clip(np.round(A_cf @ cf + rf), 0.0, mf)
    X_new[:, _JUV_MISD_IDX] = np.clip(np.round(A_cf @ cm + rm), 0.0, mm)
    X_new[:, _JUV_OTHER_IDX] = np.clip(np.round(A_cf @ co + ro), 0.0, mo)
    X_new[:, _RACE_IDX]      = black_value
    return X_new


def generate_training_race_cfs(X_train, y_train, scm, verbose=True):
    """Generate bidirectional race CFs for training set (doWhite + doBlack)."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    black_mask = X[:, _RACE_IDX] == 1
    white_mask = ~black_mask
    n_black = int(black_mask.sum())
    n_white = int(white_mask.sum())
    parts_X, parts_y = [], []
    kw = dict(cp=scm.coef_priors, cf=scm.coef_juv_fel,
              cm=scm.coef_juv_misd, co=scm.coef_juv_other,
              mp=scm.max_priors, mf=scm.max_juv_fel,
              mm=scm.max_juv_misd, mo=scm.max_juv_other)
    if n_black > 0:
        parts_X.append(_apply_intervention(
            X[black_mask],
            scm.resid_priors[black_mask], scm.resid_juv_fel[black_mask],
            scm.resid_juv_misd[black_mask], scm.resid_juv_other[black_mask],
            black_value=0.0, **kw))
        parts_y.append(y[black_mask])
    if n_white > 0:
        parts_X.append(_apply_intervention(
            X[white_mask],
            scm.resid_priors[white_mask], scm.resid_juv_fel[white_mask],
            scm.resid_juv_misd[white_mask], scm.resid_juv_other[white_mask],
            black_value=1.0, **kw))
        parts_y.append(y[white_mask])
    X_cf = np.vstack(parts_X)
    y_cf = np.concatenate(parts_y)
    if verbose:
        print(f"[CF] Race CFs: {n_black} doWhite (Black->White) + {n_white} doBlack (White->Black) = {len(X_cf):,} total")
        X_orig = np.vstack([X[black_mask], X[white_mask]])
        dp = np.abs(X_cf[:, _PRIORS_IDX] - X_orig[:, _PRIORS_IDX])
        dj = np.abs(X_cf[:, _JUV_FEL_IDX] - X_orig[:, _JUV_FEL_IDX])
        print(f"       priors_count delta:  mean={dp.mean():.3f}  max={dp.max():.3f}")
        print(f"       juv_fel_count delta: mean={dj.mean():.3f}  max={dj.max():.3f}")
    return X_cf, y_cf


def generate_test_race_cfs(X_test, scm):
    """Generate race CFs for test set (for counterfactual fairness evaluation)."""
    X = np.asarray(X_test, dtype=float)
    N = len(X)
    black  = X[:, _RACE_IDX]
    A_test = np.column_stack([np.ones(N), black])
    rp = X[:, _PRIORS_IDX]    - (A_test @ scm.coef_priors)
    rf = X[:, _JUV_FEL_IDX]  - (A_test @ scm.coef_juv_fel)
    rm = X[:, _JUV_MISD_IDX] - (A_test @ scm.coef_juv_misd)
    ro = X[:, _JUV_OTHER_IDX] - (A_test @ scm.coef_juv_other)
    black_cf = 1.0 - black
    A_cf = np.column_stack([np.ones(N), black_cf])
    X_cf = X.copy()
    X_cf[:, _PRIORS_IDX]    = np.clip(np.round(A_cf @ scm.coef_priors  + rp), 0.0, scm.max_priors)
    X_cf[:, _JUV_FEL_IDX]  = np.clip(np.round(A_cf @ scm.coef_juv_fel + rf), 0.0, scm.max_juv_fel)
    X_cf[:, _JUV_MISD_IDX] = np.clip(np.round(A_cf @ scm.coef_juv_misd + rm), 0.0, scm.max_juv_misd)
    X_cf[:, _JUV_OTHER_IDX] = np.clip(np.round(A_cf @ scm.coef_juv_other + ro), 0.0, scm.max_juv_other)
    X_cf[:, _RACE_IDX]      = black_cf
    return X_cf


# Process each dataset
def main_MM(X_train, y_train, augmentation_method, dataset_name):
    # Paths
    path_data    = os.path.join(DATA_DIR, "")          # e.g. .../New/data/
    path_results = os.path.join(OUTPUT_DIR, "results", "mm_cf_csvs", "")
    # Ensure both directories exist
    os.makedirs(path_data,    exist_ok=True)
    os.makedirs(path_results, exist_ok=True)

    # Initialize lists to store data for white and male comparators
    white_comparator_data = []
    male_comparator_data = []
    dependent_columns =  datasets_info
    sen_columns = [RACE_ENC_COL]
    # for dependent_columns in dependent_columns.items():
    #print(f"Processing dataset: {dataset_name}")

    dataset = pd.DataFrame(X_train, columns = FEATURE_COLS) #data[FEATURE_COLS].astype(float)
    label = pd.Series(y_train, name=TARGET_COL) #data[[TARGET_COL]].astype(float)

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
            #print(f"Processing dependent column: {dep_col}")

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
                # has_constant='add' forces the intercept to always be included,
                # even when a predictor column is constant (e.g. after do_white
                # sets nonwhite=0 for all rows). Without this, sm.add_constant
                # silently skips the intercept during prediction, causing a
                # shape mismatch between the design matrix and the fitted params.
                model = sm.OLS(df[dep_col], sm.add_constant(df[["female", "nonwhite", "age"]], has_constant='add')).fit()
                #print(model.summary())

                # Abduction step: estimate residuals
                df[f"resid_{dep_col}"] = df[dep_col] - model.predict(sm.add_constant(df[["female", "nonwhite", "age"]], has_constant='add'))

                # Generate counterfactuals
                for action, action_values in {"do_male": {"female": 0}, "do_white": {"nonwhite": 0}}.items():
                    cf_df = df.copy()
                    for col, value in action_values.items():
                        cf_df[col] = value
                    train_columns = ["female", "nonwhite", "age"]

                    cf_df[f"scf_{dep_col}"] = model.predict(sm.add_constant(cf_df[train_columns], has_constant='add')) + cf_df[f"resid_{dep_col}"]
                    # Round the predicted values to integers
                    cf_df[f"scf_{dep_col}"] = cf_df[f"scf_{dep_col}"].round().astype(int)

                    # Save results
                    output_file = f"{path_results}{action}_{dataset_name}_{dep_col}.csv"
                    cf_df.to_csv(output_file, index=False)
                    #print(f"Saved {action} results to {output_file}")
            else:
                # Prepare data without age
                df = dataset[[sensitive_columns[dataset_name][0], sensitive_columns[dataset_name][1], dep_col, "id"]].copy()
                df["female"] = (df[sensitive_columns[dataset_name][0]] == female_values[dataset_name]).astype(int)
                df["male"] = (df[sensitive_columns[dataset_name][0]] == male_values[dataset_name]).astype(int)
                df["nonwhite"] = (df[sensitive_columns[dataset_name][1]] != white_values[dataset_name]).astype(int)

                # Train regression model without age
                model = sm.OLS(df[dep_col], sm.add_constant(df[["female", "nonwhite"]], has_constant='add')).fit()
                #print(model.summary())

                # Abduction step: estimate residuals
                df[f"resid_{dep_col}"] = df[dep_col] - model.predict(sm.add_constant(df[["female", "nonwhite"]], has_constant='add'))

                # Generate counterfactuals
                for action, action_values in {"do_male": {"female": 0}, "do_white": {"nonwhite": 0}}.items():
                    cf_df = df.copy()
                    for col, value in action_values.items():
                        cf_df[col] = value
                    train_columns = ["female", "nonwhite"]

                    cf_df[f"scf_{dep_col}"] = model.predict(sm.add_constant(cf_df[train_columns], has_constant='add')) + cf_df[f"resid_{dep_col}"]
                    # Round the predicted values to integers
                    cf_df[f"scf_{dep_col}"] = cf_df[f"scf_{dep_col}"].round().astype(int)

                    # Save results
                    output_file = f"{path_results}{action}_{dataset_name}_{dep_col}.csv"
                    cf_df.to_csv(output_file, index=False)
                    #print(f"Saved {action} results to {output_file}")

        # After processing all dependent columns, merge and save comparator files
        # if white_comparator_data:
            # white_merged = pd.concat(white_comparator_data).drop_duplicates(subset="id")
    if PROTECTED =='RACE':
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
        #print(f"Saved white comparators to {path_results}white_comparators_{dataset_name}.csv")
                ## Decide on the augmentation method:
                ### 1. add query rows to the original dataset whit comparator lables
                ### 2. Add comparator with the oridinal label
                ### 3. 
                ###### Create augmented set by matching rows in white_comparator_data with original dataset based on common features (excluding id and target)
        
        if augmentation_method == "update_labels":
            # Build original data with IDs + targets
            original_df =  dataset[FEATURE_COLS].astype(float).copy() #fetcher.dataset["X"].copy()
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

                # AUG_RELABEL_K_NEIGHBORS (config): number of white neighbours
                # to consult per non-white row. K=1 reproduces the original
                # single-NN behaviour. K>1 lets us require neighbour agreement
                # (see AUG_RELABEL_AGREEMENT_THRESHOLD) before relabelling, so
                # one outlier neighbour cannot drive the new label.
                # None → original single-NN behaviour (K=1).
                if AUG_RELABEL_K_NEIGHBORS is None:
                    k_neighbors = 1
                else:
                    k_neighbors = max(1, int(AUG_RELABEL_K_NEIGHBORS))
                k_neighbors = min(k_neighbors, len(X_white))
                nn_white = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
                nn_white.fit(X_white)
                distances, idx_white = nn_white.kneighbors(X_nonwhite)
                # distances / idx_white have shape (n_nonwhite, k_neighbors).

                # Pick the candidate label per non-white row.
                #   - K=1 or no agreement threshold: take the single nearest
                #     neighbour's label (original behaviour).
                #   - K>1 with threshold: take the majority label among the K
                #     neighbours, but only if at least `threshold` fraction of
                #     them agree on it. Otherwise the row keeps its original
                #     label (no relabelling for ambiguous matches).
                white_targets = original_whites["target"].to_numpy()
                neighbor_labels = white_targets[idx_white]            # (n_nonwhite, K)
                if k_neighbors == 1 or AUG_RELABEL_AGREEMENT_THRESHOLD is None:
                    candidate_labels = neighbor_labels[:, 0]
                    agreement_mask   = np.ones(len(neighbor_labels), dtype=bool)
                else:
                    # Majority label and the fraction of neighbours backing it.
                    # Works for binary labels {0,1}; trivially extends if
                    # additional integer classes appear.
                    pos_frac = neighbor_labels.mean(axis=1)
                    candidate_labels = (pos_frac >= 0.5).astype(neighbor_labels.dtype)
                    majority_frac    = np.where(candidate_labels == 1, pos_frac, 1.0 - pos_frac)
                    agreement_mask   = majority_frac >= float(AUG_RELABEL_AGREEMENT_THRESHOLD)

                # AUG_RELABEL_DISTANCE_PERCENTILE (config):
                # None  → relabel all non-white rows (subject to agreement).
                # int p → also require the row's *closest* white-neighbour
                #         distance to fall within the p-th percentile of all
                #         such distances (drops far-away noisy matches).
                nearest_dist = distances[:, 0]
                if AUG_RELABEL_DISTANCE_PERCENTILE is not None:
                    threshold = np.percentile(nearest_dist, AUG_RELABEL_DISTANCE_PERCENTILE)
                    distance_mask = nearest_dist <= threshold
                else:
                    distance_mask = np.ones(len(nearest_dist), dtype=bool)

                keep_mask = agreement_mask & distance_mask
                relabel_idx    = nonwhite_idx[keep_mask]
                relabel_labels = candidate_labels[keep_mask]

                desired_targets.loc[relabel_idx] = relabel_labels

                # Diagnostics: how many rows were actually relabelled / skipped.
                n_total   = len(nonwhite_idx)
                n_kept    = int(keep_mask.sum())
                n_dropped_agree = int((~agreement_mask).sum())
                n_dropped_dist  = int(agreement_mask.sum() - keep_mask.sum())
                print(f"[update_labels] K={k_neighbors}  "
                      f"agreement_thr={AUG_RELABEL_AGREEMENT_THRESHOLD}  "
                      f"dist_pct={AUG_RELABEL_DISTANCE_PERCENTILE}  "
                      f"-> relabelled {n_kept}/{n_total} "
                      f"(skipped {n_dropped_agree} low-agreement, "
                      f"{n_dropped_dist} far-distance)")

            # Force downstream NN block to copy `desired_targets` exactly
            original_df["target"] = desired_targets.reset_index(drop=True).copy()
            
            augmented_set = original_df.copy()# pd.concat([augmented_set, white_comparator_data], ignore_index=True)

            X_aug = augmented_set[FEATURE_COLS].values
            y_aug = augmented_set['target'].values

            print("number of positive labels in augmented set:", augmented_set["target"].sum())
            print("number of positive labels in original set:", label.sum())
            augmented_set.to_csv(f"{path_results}augmented_set_{dataset_name}_{augmentation_method}.csv", index=False)
        
        elif augmentation_method == "add_comparators":
            # Add comparator rows to the original dataset
            original_df = dataset[FEATURE_COLS].astype(float).copy()
            original_df["id"] = dataset["id"].values
            original_df["target"] = label.values if hasattr(label, "values") else label

            race_col      = sensitive_columns[dataset_name][1]
            white_val     = float(white_values[dataset_name])
            id_to_target  = original_df.set_index("id")["target"]

            # ---------- direction 1: non-white -> white ----------
            # Take rows that were originally non-white, use the do(race=white)
            # intervened features (already baked into white_comparator_data),
            # and explicitly set race=white. Keep the original target.
            nonwhite_mask = white_comparator_data[race_col].astype(float) != white_val
            nonwhite_rows = white_comparator_data[nonwhite_mask].copy()
            nonwhite_rows[race_col] = white_val
            nonwhite_rows["target"] = nonwhite_rows["id"].map(id_to_target)
            nonwhite_rows = nonwhite_rows.reindex(columns=original_df.columns, fill_value=np.nan)

            # ---------- direction 2 (optional): white -> non-white ----------
            # Symmetric flip: take originally-white rows, keep their features,
            # set race=non-white. We don't have do(race=non-white) intervened
            # features, so this flip preserves features (race-only intervention).
            # Binary race assumption: non_white_value = 1 - white_value.
            if AUG_COMPARATORS_BIDIRECTIONAL:
                non_white_val = 1.0 - white_val
                white_mask    = white_comparator_data[race_col].astype(float) == white_val
                # Use original (un-intervened) features for the white side
                # so we don't double-apply do(white) to already-white rows.
                white_src = dataset.loc[white_mask.values].copy()
                white_src[race_col] = non_white_val
                white_src["target"] = white_src["id"].map(id_to_target)
                white_src = white_src.reindex(columns=original_df.columns, fill_value=np.nan)
                comp_rows = pd.concat([nonwhite_rows, white_src], ignore_index=True)
                print(f"[add_comparators] BIDIRECTIONAL: "
                      f"{len(nonwhite_rows)} non-white->white + "
                      f"{len(white_src)} white->non-white = {len(comp_rows)} total")
            else:
                comp_rows = nonwhite_rows
                print(f"[add_comparators] one-sided: {len(comp_rows)} non-white->white")

            X_aug = comp_rows[FEATURE_COLS].values
            y_aug = comp_rows["target"].values

            augmented_set = pd.concat([original_df, comp_rows], ignore_index=True)
            print("number of positive labels in augmented set:", augmented_set["target"].sum())
            print("number of positive labels in original set:", label.sum())
            augmented_set.to_csv(
                f"{path_results}augmented_set_white_{dataset_name}_{augmentation_method}.csv",
                index=False,
            )

    # NOTE: the PROTECTED == 'SEX' branch used to live here but was truncated
    # by a filesystem-sync incident. It is not reachable with the current
    # config (PROTECTED='RACE'), so we simply return the RACE-branch outputs.
    return X_aug, y_aug
