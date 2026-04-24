from __future__ import annotations
import os, sys, warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import (
    COMPAS_CSV, FEATURE_COLS, TARGET_COL, RACE_ENC_COL, SEX_ENC_COL,
    CHARGE_DEGREE_ENC_COL, PROTECTED_COL, RANDOM_STATE, TEST_SIZE, VAL_SIZE,
    OUTPUT_DIR, DATASET_NAME, AUGMENTATION_METHOD, SCALED,
    AUG_DROP_IDENTICAL_CFS, AUG_SYNTHETIC_WEIGHT,
)
from pipeline.cf_generation import SCMParams, fit_scm, generate_training_race_cfs, main_MM

_MM_CF_DIR = os.path.join(OUTPUT_DIR, "results", "mm_cf_arrays")


@dataclass
class DataBundle:
    X_train:  np.ndarray = field(default_factory=lambda: np.array([]))
    y_train:  np.ndarray = field(default_factory=lambda: np.array([]))
    X_val:    np.ndarray = field(default_factory=lambda: np.array([]))
    y_val:    np.ndarray = field(default_factory=lambda: np.array([]))
    X_test:   np.ndarray = field(default_factory=lambda: np.array([]))
    y_test:   np.ndarray = field(default_factory=lambda: np.array([]))
    X_cf_race:   np.ndarray = field(default_factory=lambda: np.array([]))
    y_cf_race:   np.ndarray = field(default_factory=lambda: np.array([]))
    X_aug:    np.ndarray = field(default_factory=lambda: np.array([]))
    y_aug:    np.ndarray = field(default_factory=lambda: np.array([]))
    X_train_sc:   np.ndarray = field(default_factory=lambda: np.array([]))
    X_val_sc:     np.ndarray = field(default_factory=lambda: np.array([]))
    X_test_sc:    np.ndarray = field(default_factory=lambda: np.array([]))
    X_cf_race_sc: np.ndarray = field(default_factory=lambda: np.array([]))
    X_aug_sc:     np.ndarray = field(default_factory=lambda: np.array([]))
    train_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    full_df:      Optional[pd.DataFrame] = None
    scaler:       Optional[MinMaxScaler] = None
    scm:          Optional[SCMParams]    = None
    feature_cols: list = field(default_factory=list)


def augment_data(X_train: np.ndarray, y_train: np.ndarray,
                 method: str, scm,
                 dataset_tag: str = DATASET_NAME.lower(),
                 verbose: bool = True):
    """Return (X_aug, y_aug, sample_weight) for *method* applied to X_train / y_train.

    Single source of truth for augmentation shared between the initial training
    phase and the LDP phase in unified_analysis.py.

    Parameters
    ----------
    X_train     : raw (unscaled) training features
    y_train     : training labels
    method      : 'SCM' | 'update_labels' | 'add_comparators'
    scm         : fitted SCMParams (required for method='SCM')
    dataset_tag : dataset key for main_MM config lookups ('compas' / 'adult')
    verbose     : passed through to generate_training_race_cfs

    Returns
    -------
    X_aug         : augmented feature matrix (original + synthetic rows)
    y_aug         : augmented labels
    sample_weight : per-row weight array, or None if AUG_SYNTHETIC_WEIGHT is None.
                    Original rows have weight 1.0; synthetic rows have weight
                    AUG_SYNTHETIC_WEIGHT (set in config.py).

    Config flags applied here (all default to original behaviour):
      AUG_DROP_IDENTICAL_CFS      — drop synthetic rows identical to source
      AUG_SYNTHETIC_WEIGHT        — down-weight synthetic rows during training
    AUG_RELABEL_DISTANCE_PERCENTILE is applied inside main_MM (cf_generation.py).
    """
    n_orig = len(X_train)

    if method == "SCM":
        if verbose:
            print("[data] Generating SCM race counterfactuals ...")
        X_cf, y_cf = generate_training_race_cfs(X_train, y_train, scm, verbose=verbose)

    elif method == "update_labels":
        if verbose:
            print("[data] Running MM - update_labels ...")
        # update_labels returns a relabelled version of X_train (same rows,
        # updated labels) — no new rows are appended here.
        X_aug, y_aug = main_MM(X_train, y_train, "update_labels", dataset_tag)
        # No synthetic rows to weight or filter: return directly.
        sample_weight = None  # uniform (update_labels doesn't append rows)
        if verbose:
            n_changed = int((y_aug != y_train).sum())
            print(f"[data] Labels changed: {n_changed}/{n_orig} "
                  f"({n_changed/n_orig*100:.1f}%)")
            
        #X_aug = np.vstack([X_train, X_aug])
        #y_aug = np.concatenate([y_train, y_aug])

        return X_aug, y_aug, sample_weight

    else:  # add_comparators
        if verbose:
            print("[data] Running MM - add_comparators ...")
        X_cf, y_cf = main_MM(X_train, y_train, "add_comparators", dataset_tag)

    # --- apply AUG_DROP_IDENTICAL_CFS ---
    # Synthetic rows that are bit-for-bit identical to their source add noise
    # without new information. Drop them when the flag is enabled.
    if AUG_DROP_IDENTICAL_CFS and len(X_cf) > 0:
        # For SCM / add_comparators X_cf aligns row-for-row with X_train
        # (each CF was generated from the corresponding training row).
        n_cf = len(X_cf)
        n_src = min(n_cf, n_orig)
        diff_mask = np.any(X_cf[:n_src] != X_train[:n_src], axis=1)
        if n_cf > n_src:
            # extra CFs beyond X_train length — keep all of those
            diff_mask = np.concatenate([diff_mask, np.ones(n_cf - n_src, dtype=bool)])
        n_dropped = int((~diff_mask).sum())
        if n_dropped > 0 and verbose:
            print(f"[data] AUG_DROP_IDENTICAL_CFS: dropped {n_dropped}/{n_cf} identical CFs")
        X_cf = X_cf[diff_mask]
        y_cf = y_cf[diff_mask]

    X_aug = np.vstack([X_train, X_cf])
    y_aug = np.concatenate([y_train, y_cf])

    # --- build sample_weight ---
    # Original rows get weight 1.0; synthetic rows get AUG_SYNTHETIC_WEIGHT.
    # None means uniform weights (sklearn default — no weight array passed).
    if AUG_SYNTHETIC_WEIGHT is not None:
        sample_weight = np.ones(len(X_aug), dtype=float)
        sample_weight[n_orig:] = float(AUG_SYNTHETIC_WEIGHT)
        if verbose:
            print(f"[data] Sample weights: original=1.0  synthetic={AUG_SYNTHETIC_WEIGHT}")
    else:
        sample_weight = None

    return X_aug, y_aug, sample_weight


def _encode_compas(df):
    df = df.copy()
    df[RACE_ENC_COL]          = (df["race"] == "African-American").astype(int)
    df[SEX_ENC_COL]           = (df["sex"].str.lower() == "female").astype(int)
    df[CHARGE_DEGREE_ENC_COL] = (df["c_charge_degree"] == "F").astype(int)
    df[TARGET_COL]            = df["two_year_recid"].astype(int)
    return df


def _load_compas():
    if not os.path.isfile(COMPAS_CSV):
        raise FileNotFoundError(f"COMPAS CSV not found at: {COMPAS_CSV}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(COMPAS_CSV)
    df = df[
        df["days_b_screening_arrest"].notna() &
        (df["days_b_screening_arrest"].abs() <= 30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O")
    ].copy()

    #df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    df = df.copy()
    white_groups = ["Caucasian", "Hispanic", "Other"]

    df["race"] = df["race"].apply(
        lambda x: "White" if x in white_groups else "African-American"
    )
    
    df = _encode_compas(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def load_and_split(augmentation_method: str = AUGMENTATION_METHOD,
                   output_dir: str = None,
                   verbose: bool = True):
    """Load COMPAS, split, fit SCM and build augmented training set.

    Parameters
    ----------
    augmentation_method : one of 'SCM', 'update_labels', 'add_comparators'
    output_dir          : where to cache MM-CF arrays (defaults to OUTPUT_DIR)
    verbose             : print progress
    """
    df = _load_compas()
    if verbose:
        print(f"[data] Loaded {len(df):,} rows from compas-scores-two-years.csv")
        print(f"       recid=1  {df[TARGET_COL].mean()*100:.1f}%")
        print(f"       Black    {df[RACE_ENC_COL].mean()*100:.1f}%")
        print(f"       Female   {df[SEX_ENC_COL].mean()*100:.1f}%")

    X_all = df[FEATURE_COLS].values.astype(float)
    y_all = df[TARGET_COL].values.astype(int)
    strat = df[TARGET_COL].astype(str) + "_" + df[RACE_ENC_COL].astype(str)

    X_tv, X_test, y_tv, y_test, idx_tv, _ = train_test_split(
        X_all, y_all, np.arange(len(df)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=strat)

    strat_tv = strat.values[idx_tv]
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE) if VAL_SIZE > 0 else 0.0

    if val_frac > 0:
        X_train, X_val, y_train, y_val, idx_train, _ = train_test_split(
            X_tv, y_tv, idx_tv, test_size=val_frac,
            random_state=RANDOM_STATE, stratify=strat_tv)
    else:
        X_train, y_train, idx_train = X_tv, y_tv, idx_tv
        X_val   = np.empty((0, len(FEATURE_COLS)), dtype=float)
        y_val   = np.empty(0, dtype=int)

    if verbose:
        print(f"[data] Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

    scaler = MinMaxScaler() #StandardScaler() #MinMaxScaler()
    scaler.fit(X_train)

    def sc(X):
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return np.empty((0, len(FEATURE_COLS)), dtype=float)
        return scaler.transform(arr)

    if SCALED:
        X_train=sc(X_train)
        X_test=sc(X_test)
        X_val=sc(X_val)

    # SCM is always fitted — used for CF-fairness evaluation on the test set
    # regardless of which augmentation method is used for training.
    X_cf_race = np.empty((0, len(FEATURE_COLS)), dtype=float)
    y_cf_race = np.empty(0, dtype=int)


    scm = fit_scm(X_train, verbose=verbose)

    if verbose:
        print(f"[data] Building augmented set: {augmentation_method} ...")

    X_aug, y_aug, _ = augment_data(X_train, y_train, augmentation_method, scm, dataset_tag=DATASET_NAME.lower(), verbose=verbose)

    # Keep X_cf_race / y_cf_race for the DataBundle fields.
    # For SCM: the CFs are X_aug minus X_train.
    # For update_labels: there are no appended CFs (labels are updated in-place).
    # For add_comparators: the CFs are X_aug minus X_train.
    if augmentation_method == 'SCM' or augmentation_method == 'add_comparators':
        X_cf_race = X_aug[len(X_train):]
        y_cf_race = y_aug[len(y_train):]
    # else update_labels: X_cf_race stays as the empty arrays initialised above


    if verbose:
        print(f"[data] Augmented training set: {len(X_aug):,} rows ")

    # Persist MM-CF arrays for reuse — one subfolder per augmentation method
    _out = output_dir or OUTPUT_DIR
    _cache_dir = os.path.join(_out, "results", "mm_cf_arrays", augmentation_method)
    try:
        os.makedirs(_cache_dir, exist_ok=True)
        np.save(os.path.join(_cache_dir, "X_cf_race.npy"), X_cf_race)
        np.save(os.path.join(_cache_dir, "y_cf_race.npy"), y_cf_race)
        np.save(os.path.join(_cache_dir, "X_aug.npy"),     X_aug)
        np.save(os.path.join(_cache_dir, "y_aug.npy"),     y_aug)
        np.save(os.path.join(_cache_dir, "X_train.npy"),   X_train)
        np.save(os.path.join(_cache_dir, "y_train.npy"),   y_train)
        np.save(os.path.join(_cache_dir, "X_test.npy"),    X_test)
        np.save(os.path.join(_cache_dir, "y_test.npy"),    y_test)
        if verbose:
            print(f"[data] MM-CF arrays saved -> {_cache_dir}")
    except Exception as e:
        warnings.warn(f"[data] Could not save MM-CF arrays: {e}")

    # sc are same as input (made on purpose) - to be edited if needed
    return DataBundle(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_cf_race=X_cf_race, y_cf_race=y_cf_race,
        X_aug=X_aug, y_aug=y_aug,
        X_train_sc=X_train, X_val_sc=X_val,
        X_test_sc=X_test, X_cf_race_sc=X_cf_race,
        X_aug_sc=sc(X_aug),
        train_indices=idx_train,
        full_df=df, scaler=scaler, scm=scm,
        feature_cols=FEATURE_COLS,
    )