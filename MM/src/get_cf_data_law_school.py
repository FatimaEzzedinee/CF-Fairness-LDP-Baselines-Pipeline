"""Generate structural counterfactual data for supported datasets."""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    file_name: str
    output_prefix: str
    sep: str = "|"
    race_col: str = "race_nonwhite"
    sex_col: str = "sex"
    lsat_col: str = "LSAT"
    ugpa_col: str = "UGPA"
    female_value: object = "Female"
    male_value: object = "Male"
    white_value: object = "White"
    nonwhite_value: object = "NonWhite"
    lsat_min: float = 10.0
    lsat_max: float = 48.0
    ugpa_min: float = 0.0
    ugpa_max: float = 4.0
    target_col: str | None = None
    positive_value: object = 1
    resid_col_1: str = "resid_LSAT"
    resid_col_2: str = "resid_UGPA"
    scf_col_1: str = "scf_LSAT"
    scf_col_2: str = "scf_UGPA"
    output_sex_col: str = "Sex"
    output_race_col: str = "Race"
    target_type: str = "binary"
    recalc_target_only: bool = False
    target_resid_col: str = "resid_target"
    target_scf_col: str = "scf_target"


SUPPORTED_DATASETS = {
    "lawschool": DatasetConfig(
        key="lawschool",
        file_name="LawSchool.csv",
        output_prefix="LawSchool",
    ),
    "adult": DatasetConfig(
        key="adult",
        file_name="adult.csv",
        output_prefix="Adult",
        sep=",",
        race_col="race",
        sex_col="sex",
        lsat_col="education-num",
        ugpa_col="hours-per-week",
        female_value=0,
        male_value=1,
        white_value=4,
        nonwhite_value=None,
        lsat_min=1.0,
        lsat_max=16.0,
        ugpa_min=1.0,
        ugpa_max=99.0,
        target_col="target",
        positive_value=1,
        resid_col_1="resid_education_num",
        resid_col_2="resid_hours_per_week",
        scf_col_1="scf_education_num",
        scf_col_2="scf_hours_per_week",
        output_sex_col="sex",
        output_race_col="race",
        target_type="binary",
    ),
    "folktables": DatasetConfig(
        key="folktables",
        file_name="folktables.csv",
        output_prefix="Folktables",
        sep=",",
        race_col="race",
        sex_col="gender",
        lsat_col="education-num",
        ugpa_col="hours-per-week",
        female_value="Female",
        male_value="Male",
        white_value="White",
        nonwhite_value=None,
        lsat_min=1.0,
        lsat_max=16.0,
        ugpa_min=1.0,
        ugpa_max=99.0,
        target_col="income",
        target_type="continuous",
        recalc_target_only=True,
        target_resid_col="resid_income",
        target_scf_col="scf_income",
        resid_col_1="resid_education_num",
        resid_col_2="resid_hours_per_week",
        scf_col_1="scf_education_num",
        scf_col_2="scf_hours_per_week",
        output_sex_col="gender",
        output_race_col="race",
    ),
    "lawschool_debiased": DatasetConfig(
        key="lawschool_debiased",
        file_name="debiased_LawSchool_data.csv",
        output_prefix="LawSchoolDebiased",
    ),
}


def _build_sensitive_columns(df, config):
    """Build binary sensitive columns from dataset-specific encodings."""
    df["female"] = (df[config.sex_col] == config.female_value).astype(int)
    df["male"] = (df[config.sex_col] == config.male_value).astype(int)
    df["white"] = (df[config.race_col] == config.white_value).astype(int)
    if config.nonwhite_value is None:
        df["nonwhite"] = (df[config.race_col] != config.white_value).astype(int)
    else:
        df["nonwhite"] = (df[config.race_col] == config.nonwhite_value).astype(int)


def fetch_dataset(dataset_name, data_dir):
    """Read and preprocess a selected dataset and return all required objects."""
    key = dataset_name.strip().lower()
    if key not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS.keys()))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose one of: {supported}")

    config = SUPPORTED_DATASETS[key]
    dataset_path = os.path.join(data_dir, config.file_name)
    org_df = pd.read_csv(dataset_path, sep=config.sep)

    vars_base = [config.lsat_col, config.ugpa_col, config.sex_col, config.race_col]
    if config.target_col is not None:
        vars_base.append(config.target_col)
    missing_cols = [c for c in vars_base if c not in org_df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Dataset '{dataset_name}' is missing required columns: {missing}")

    df = org_df[vars_base].copy()
    df[config.lsat_col] = df[config.lsat_col].round()
    if config.target_col is not None and config.target_type == "continuous":
        df[config.target_col] = pd.to_numeric(df[config.target_col], errors="coerce")

    sense_cols = ["female", "male", "white", "nonwhite"]
    _build_sensitive_columns(df, config)

    vars_m = [config.lsat_col, config.ugpa_col] + sense_cols

    return {
        "config": config,
        "org_df": org_df,
        "df": df,
        "vars": vars_base,
        "vars_m": vars_m,
        "sense_cols": sense_cols,
    }


def generate_level3_counterfactuals(dataset_objects, result_dir):
    """Generate level-3 structural counterfactuals for gender and race interventions."""
    config = dataset_objects["config"]
    df = dataset_objects["df"]
    df_lev3 = df.copy()

    predictors = ["female", "nonwhite"]

    df_lev3_do_male = pd.DataFrame(
        {
            "female": np.zeros(len(df_lev3), dtype=int),
            "nonwhite": df_lev3["nonwhite"],
        }
    )

    df_lev3_do_white = pd.DataFrame(
        {
            "female": df_lev3["female"],
            "nonwhite": np.zeros(len(df_lev3), dtype=int),
        }
    )

    if config.recalc_target_only:
        if config.target_col is None:
            raise ValueError(f"Dataset '{config.key}' requires target_col when recalc_target_only=True")

        model_target = sm.OLS(df_lev3[config.target_col], sm.add_constant(df_lev3[predictors])).fit()
        df_lev3[config.target_resid_col] = df_lev3[config.target_col] - model_target.predict(
            sm.add_constant(df_lev3[predictors])
        )

        target_min = float(df_lev3[config.target_col].min())
        target_max = float(df_lev3[config.target_col].max())

        for do_df in (df_lev3_do_male, df_lev3_do_white):
            do_df[config.output_sex_col] = df_lev3[config.sex_col]
            do_df[config.output_race_col] = df_lev3[config.race_col]
            do_df[config.target_resid_col] = df_lev3[config.target_resid_col]
            do_df[config.target_scf_col] = (
                model_target.predict(sm.add_constant(do_df[predictors])) + do_df[config.target_resid_col]
            ).round(3)
            do_df[config.target_scf_col] = np.clip(do_df[config.target_scf_col], target_min, target_max)
    else:
        model_ugpa = sm.OLS(df_lev3[config.ugpa_col], sm.add_constant(df_lev3[predictors])).fit()
        model_lsat = sm.OLS(df_lev3[config.lsat_col], sm.add_constant(df_lev3[predictors])).fit()

        df_lev3[config.resid_col_2] = df_lev3[config.ugpa_col] - model_ugpa.predict(
            sm.add_constant(df_lev3[predictors])
        )
        df_lev3[config.resid_col_1] = df_lev3[config.lsat_col] - model_lsat.predict(
            sm.add_constant(df_lev3[predictors])
        )

        for do_df in (df_lev3_do_male, df_lev3_do_white):
            do_df[config.output_sex_col] = df_lev3[config.sex_col]
            do_df[config.output_race_col] = df_lev3[config.race_col]
            do_df[config.resid_col_1] = df_lev3[config.resid_col_1]
            do_df[config.resid_col_2] = df_lev3[config.resid_col_2]
            do_df[config.scf_col_1] = (
                model_lsat.predict(sm.add_constant(do_df[predictors])) + do_df[config.resid_col_1]
            ).round(3)
            do_df[config.scf_col_2] = (
                model_ugpa.predict(sm.add_constant(do_df[predictors])) + do_df[config.resid_col_2]
            ).round(3)
            do_df[config.scf_col_1] = np.clip(do_df[config.scf_col_1], config.lsat_min, config.lsat_max)
            do_df[config.scf_col_2] = np.clip(do_df[config.scf_col_2], config.ugpa_min, config.ugpa_max)

    male_out = os.path.join(result_dir, f"cf_{config.output_prefix}_lev3_doMale.csv")
    white_out = os.path.join(result_dir, f"cf_{config.output_prefix}_lev3_doWhite.csv")
    df_lev3_do_male.to_csv(male_out, sep="|", index=False)
    df_lev3_do_white.to_csv(white_out, sep="|", index=False)

    return df_lev3_do_male, df_lev3_do_white


def debias_model(dataset_objects, df_lev3_do_male, df_lev3_do_white, result_dir):
    """Augment with counterfactual features and save a debiased training dataset."""
    config = dataset_objects["config"]
    df = dataset_objects["df"].copy()

    if config.recalc_target_only:
        if config.target_col is None:
            raise ValueError(f"Dataset '{config.key}' requires target_col when recalc_target_only=True")

        df[f"cf_{config.target_col}_male"] = df_lev3_do_male[config.target_scf_col]
        df[f"cf_{config.target_col}_white"] = df_lev3_do_white[config.target_scf_col]

        cf_male_train = df_lev3_do_male[["female", "nonwhite", config.target_scf_col]].rename(
            columns={config.target_scf_col: config.target_col}
        )
        cf_white_train = df_lev3_do_white[["female", "nonwhite", config.target_scf_col]].rename(
            columns={config.target_scf_col: config.target_col}
        )
        train_cols = [config.target_col, "female", "nonwhite"]
        factual_train = df[train_cols].copy()
        new_data = pd.concat([factual_train, cf_male_train, cf_white_train], axis=0, ignore_index=True)

        model_target_debiased = sm.OLS(
            new_data[config.target_col], sm.add_constant(new_data[["female", "nonwhite"]])
        ).fit()
        print(f"Debiased {config.target_col} Model Summary:")
        print(model_target_debiased.summary())
    else:
        df[f"cf_{config.lsat_col}_male"] = df_lev3_do_male[config.scf_col_1]
        df[f"cf_{config.ugpa_col}_male"] = df_lev3_do_male[config.scf_col_2]
        df[f"cf_{config.lsat_col}_white"] = df_lev3_do_white[config.scf_col_1]
        df[f"cf_{config.ugpa_col}_white"] = df_lev3_do_white[config.scf_col_2]

        cf_male_train = df_lev3_do_male[["female", "nonwhite", config.scf_col_1, config.scf_col_2]].rename(
            columns={config.scf_col_1: config.lsat_col, config.scf_col_2: config.ugpa_col}
        )
        cf_white_train = df_lev3_do_white[["female", "nonwhite", config.scf_col_1, config.scf_col_2]].rename(
            columns={config.scf_col_1: config.lsat_col, config.scf_col_2: config.ugpa_col}
        )
        train_cols = [config.lsat_col, config.ugpa_col, "female", "nonwhite"]
        factual_train = df[train_cols].copy()
        new_data = pd.concat([factual_train, cf_male_train, cf_white_train], axis=0, ignore_index=True)

        model_ugpa_debiased = sm.OLS(
            new_data[config.ugpa_col], sm.add_constant(new_data[["female", "nonwhite"]])
        ).fit()
        model_lsat_debiased = sm.OLS(
            new_data[config.lsat_col], sm.add_constant(new_data[["female", "nonwhite"]])
        ).fit()

        print(f"Debiased {config.ugpa_col} Model Summary:")
        print(model_ugpa_debiased.summary())
        print(f"Debiased {config.lsat_col} Model Summary:")
        print(model_lsat_debiased.summary())

    debiased_out = os.path.join(result_dir, f"debiased_{config.output_prefix}_data.csv")
    new_data.to_csv(debiased_out, sep="|", index=False)


def check_fairness(dataset_objects, df_lev3_do_male, df_lev3_do_white, result_dir):
    """Compute fairness metrics using dataset-native target and counterfactual predictions."""
    config = dataset_objects["config"]
    if config.target_col is None:
        print(f"Skipping fairness check for {config.key}: no target column configured.")
        return

    factual = dataset_objects["df"].copy()
    if config.target_col not in factual.columns:
        print(f"Skipping fairness check for {config.key}: missing target column '{config.target_col}'.")
        return

    if config.recalc_target_only:
        if config.target_col is None:
            raise ValueError(f"Dataset '{config.key}' requires target_col when recalc_target_only=True")
        y_true = pd.to_numeric(factual[config.target_col], errors="coerce")
        valid = factual[[config.target_col, "female", "male", "white", "nonwhite"]].notna().all(axis=1)
        pred_factual = pd.Series(y_true.loc[valid], index=factual.index[valid])
        pred_cf_male = pd.Series(df_lev3_do_male.loc[valid, config.target_scf_col], index=factual.index[valid])
        pred_cf_white = pd.Series(df_lev3_do_white.loc[valid, config.target_scf_col], index=factual.index[valid])
    else:
        X_train = sm.add_constant(factual[[config.lsat_col, config.ugpa_col]], has_constant="add")
        X_cf_male = sm.add_constant(
            df_lev3_do_male[[config.scf_col_1, config.scf_col_2]].rename(
                columns={config.scf_col_1: config.lsat_col, config.scf_col_2: config.ugpa_col}
            ),
            has_constant="add",
        )
        X_cf_white = sm.add_constant(
            df_lev3_do_white[[config.scf_col_1, config.scf_col_2]].rename(
                columns={config.scf_col_1: config.lsat_col, config.scf_col_2: config.ugpa_col}
            ),
            has_constant="add",
        )

        valid = factual[[config.lsat_col, config.ugpa_col, "female", "male", "white", "nonwhite"]].notna().all(axis=1)

        if config.target_type == "continuous":
            y_true = pd.to_numeric(factual[config.target_col], errors="coerce")
            valid = valid & y_true.notna()
            y_fit = y_true.loc[valid]
            outcome_model = sm.OLS(y_fit, X_train.loc[valid]).fit()

            pred_factual = pd.Series(outcome_model.predict(X_train.loc[valid]), index=factual.index[valid])
            pred_cf_male = pd.Series(outcome_model.predict(X_cf_male.loc[valid]), index=factual.index[valid])
            pred_cf_white = pd.Series(outcome_model.predict(X_cf_white.loc[valid]), index=factual.index[valid])
        else:
            y_true = (factual[config.target_col] == config.positive_value).astype(int)
            valid = valid & y_true.notna()
            outcome_model = sm.Logit(y_true.loc[valid], X_train.loc[valid]).fit(disp=False)

            pred_factual = pd.Series(
                (outcome_model.predict(X_train.loc[valid]) >= 0.5).astype(int),
                index=factual.index[valid],
            )
            pred_cf_male = pd.Series(
                (outcome_model.predict(X_cf_male.loc[valid]) >= 0.5).astype(int),
                index=factual.index[valid],
            )
            pred_cf_white = pd.Series(
                (outcome_model.predict(X_cf_white.loc[valid]) >= 0.5).astype(int),
                index=factual.index[valid],
            )

    is_female = (factual["female"] == 1) & valid
    is_male = (factual["male"] == 1) & valid
    is_nonwhite = (factual["nonwhite"] == 1) & valid
    is_white = (factual["white"] == 1) & valid

    female_idx = factual.index[is_female]
    male_idx = factual.index[is_male]
    nonwhite_idx = factual.index[is_nonwhite]
    white_idx = factual.index[is_white]

    if config.target_type == "continuous":
        metrics = [
            ("factual_mean_pred_target_female", float(pred_factual.loc[female_idx].mean())),
            ("factual_mean_pred_target_male", float(pred_factual.loc[male_idx].mean())),
            (
                "factual_gap_male_minus_female",
                float(pred_factual.loc[male_idx].mean() - pred_factual.loc[female_idx].mean()),
            ),
            ("factual_mean_pred_target_nonwhite", float(pred_factual.loc[nonwhite_idx].mean())),
            ("factual_mean_pred_target_white", float(pred_factual.loc[white_idx].mean())),
            (
                "factual_gap_white_minus_nonwhite",
                float(pred_factual.loc[white_idx].mean() - pred_factual.loc[nonwhite_idx].mean()),
            ),
            (
                "counterfactual_mean_delta_do_male_for_females",
                float((pred_cf_male.loc[female_idx] - pred_factual.loc[female_idx]).mean()),
            ),
            (
                "counterfactual_mean_delta_do_white_for_nonwhite",
                float((pred_cf_white.loc[nonwhite_idx] - pred_factual.loc[nonwhite_idx]).mean()),
            ),
            (
                "counterfactual_negative_delta_rate_do_male_for_females",
                float((pred_cf_male.loc[female_idx] < pred_factual.loc[female_idx]).mean()),
            ),
            (
                "counterfactual_negative_delta_rate_do_white_for_nonwhite",
                float((pred_cf_white.loc[nonwhite_idx] < pred_factual.loc[nonwhite_idx]).mean()),
            ),
        ]
    else:
        metrics = [
            ("factual_positive_rate_female", float(pred_factual.loc[female_idx].mean())),
            ("factual_positive_rate_male", float(pred_factual.loc[male_idx].mean())),
            (
                "factual_gap_male_minus_female",
                float(pred_factual.loc[male_idx].mean() - pred_factual.loc[female_idx].mean()),
            ),
            ("factual_positive_rate_nonwhite", float(pred_factual.loc[nonwhite_idx].mean())),
            ("factual_positive_rate_white", float(pred_factual.loc[white_idx].mean())),
            (
                "factual_gap_white_minus_nonwhite",
                float(pred_factual.loc[white_idx].mean() - pred_factual.loc[nonwhite_idx].mean()),
            ),
            (
                "counterfactual_flip_rate_do_male_for_females",
                float((pred_cf_male.loc[female_idx] != pred_factual.loc[female_idx]).mean()),
            ),
            (
                "counterfactual_flip_rate_do_white_for_nonwhite",
                float((pred_cf_white.loc[nonwhite_idx] != pred_factual.loc[nonwhite_idx]).mean()),
            ),
            (
                "counterfactual_negative_flip_rate_do_male_for_females",
                float(((pred_factual.loc[female_idx] == 1) & (pred_cf_male.loc[female_idx] == 0)).mean()),
            ),
            (
                "counterfactual_negative_flip_rate_do_white_for_nonwhite",
                float(((pred_factual.loc[nonwhite_idx] == 1) & (pred_cf_white.loc[nonwhite_idx] == 0)).mean()),
            ),
        ]

    fairness_df = pd.DataFrame(metrics, columns=["metric", "value"])
    fairness_path = os.path.join(result_dir, f"fairness_{config.output_prefix}.csv")
    fairness_df.to_csv(fairness_path, index=False)
    print(f"Saved fairness report: {fairness_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate level-3 counterfactual data for supported datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=sorted(SUPPORTED_DATASETS.keys()),
        help="Dataset key to process",
    )
    parser.add_argument("--rseed", type=int, default=4, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.rseed)

    path_data = "./data"
    path_rslt = "./data"

    dataset_objects = fetch_dataset(args.dataset, path_data)
    df_lev3_do_male, df_lev3_do_white = generate_level3_counterfactuals(
        dataset_objects, path_rslt
    )
    debias_model(dataset_objects, df_lev3_do_male, df_lev3_do_white, path_rslt)
    check_fairness(dataset_objects, df_lev3_do_male, df_lev3_do_white, path_rslt)


if __name__ == "__main__":
    main()