"""
unified_analysis.py  --  Structured 8-step fairness/privacy pipeline.

Execution order
---------------
FAIRNESS BLOCK
  Step 1  Baseline          : LR / RF / XGB trained on raw X_train
  Step 2  Augmented         : one model set per augmentation method
                              (SCM, update_labels, add_comparators)
  Step 3  Fairness analysis : group + individual + CF fairness  (Steps 1-2)
  Step 4  Fairlearn         : all constraint variants trained on X_train
  Step 5  Full fairness     : compare everything so far (1 + 2 + 4)
                              → save step5_fairness_all_no_ldp.csv

PRIVACY BLOCK
  Step 6  LDP               : randomised-response on X_train → X_train_ldp
                              then RE-augment X_train_ldp with every method
                              (LDP-Baseline, LDP-SCM, LDP-update_labels,
                               LDP-add_comparators, LDP-Fairlearn)
  Step 7  Privacy + fairness: compare all previous + LDP variants
                              → save step7_fairness_all.csv

    Step 8  AIA / NiCE CFs   : full sensitive-attribute inference on key scenarios
                                                            → save step8_aia.csv + step8_nice_quality.csv

All CSVs land in  pipeline_outputs/unified/results/

Usage
-----
  python pipeline/unified_analysis.py                           # full run
    python pipeline/unified_analysis.py --skip-aia --skip-nice   # fairness only
  python pipeline/unified_analysis.py --methods SCM update_labels
  python pipeline/unified_analysis.py --ldp-epsilons 0.5 1.0
  python pipeline/unified_analysis.py --families lr rf
  python pipeline/unified_analysis.py --no-fairlearn
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR  = os.path.dirname(_THIS_DIR)
for _p in [_BASE_DIR, os.path.join(_BASE_DIR, "ensemble_mia-main")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.config import (
    OUTPUT_DIR, MODEL_FAMILIES, LDP_EPSILONS, NICE_MAX_SAMPLES,
    AUGMENTATION_METHOD, DATASET_NAME, FEATURE_COLS,
    PROTECTED_COL, RANDOM_STATE,
)
import pipeline.data_preparation as dp
import pipeline.models            as ml
import pipeline.fairness_metrics  as fm
import pipeline.ldp               as ldp_mod
import pipeline.mia_analysis      as mia_mod
import pipeline.attribute_inference_attack as aia_mod
import pipeline.nice_cf           as ncf
from pipeline.data_preparation import augment_data

_PROTECTED_IDX = FEATURE_COLS.index(PROTECTED_COL)

ALL_AUG_METHODS   = ["SCM", "update_labels", "add_comparators", "add_comparators_bidir"]
ATTACKS            = ["mia", "aia"]  # membership inference attack, attribute inference attack
# "add_comparators_bidir" is a virtual method: it runs add_comparators with
# the AUG_COMPARATORS_BIDIRECTIONAL config flag forced to True for that
# scenario only (the one-sided "add_comparators" run forces it to False).
_UNIFIED_DIR      = os.path.join(OUTPUT_DIR, "unified")
_UNIFIED_RES      = os.path.join(_UNIFIED_DIR, "results")
_NICE_CACHE_DIR   = os.path.join(_UNIFIED_DIR, "nice_cf_arrays")
_DEFAULT_EPSILONS = LDP_EPSILONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    b = "=" * 60
    print(f"\n{b}\n  {text}\n{b}")


def _mkdirs() -> None:
    for d in [_UNIFIED_DIR, _UNIFIED_RES, _NICE_CACHE_DIR]:
        os.makedirs(d, exist_ok=True)


def _save_csv(df: pd.DataFrame, fname: str) -> str:
    path = os.path.join(_UNIFIED_RES, fname)
    df.to_csv(path, index=False)
    print(f"  [save] {fname}  ({len(df)} rows)  →  {path}")
    return path


# ---------------------------------------------------------------------------
# Fairness extraction helper
# ---------------------------------------------------------------------------

def _fairness_rows(fairness_dict: dict, extra_cols: dict = None) -> list:
    """Flatten {scenario: {family: FairnessResult}} → list of dicts."""
    rows = []
    for sc, families in fairness_dict.items():
        for fam, fr in families.items():
            row = {"scenario": sc, "family": fam, "model": fr.model_name}
            row.update({f"group_{k}": v for k, v in fr.group.items()})
            row.update({f"ind_{k}": v   for k, v in fr.individual.items()})
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    return rows


def _metrics_rows(scenarios: dict, extra_cols: dict = None) -> list:
    """Flatten ScenarioModels dict → list of model-metric dicts."""
    rows = []
    for sc, sm in scenarios.items():
        for fam, res in sm.results.items():
            m = res.metrics
            if not m:
                continue
            row = {
                "scenario":  sc, "family": fam, "model": res.name,
                "n_train":   res.n_train,
                "accuracy":  m.get("accuracy",  float("nan")),
                "auc_roc":   m.get("auc_roc",   float("nan")),
                "f1":        m.get("f1",         float("nan")),
                "precision": m.get("precision",  float("nan")),
                "recall":    m.get("recall",     float("nan")),
            }
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    return rows


def _nice_rows(nice_results: dict, extra_cols: dict = None) -> list:
    rows = []
    for sc, families in nice_results.items():
        for fam, nr in families.items():
            if not getattr(nr, "metrics", {}):
                continue
            row = {"scenario": sc, "family": fam, "model": nr.model_name}
            yq = getattr(nr, "y_query", np.array([]))
            yc = getattr(nr, "y_cf",    np.array([]))
            if len(yq) > 0 and len(yc) > 0:
                row["flip_rate"] = float((yq != yc).mean())
            row.update(nr.metrics)
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Fairlearn helper  — wraps train_fair_models into ScenarioModels format
# ---------------------------------------------------------------------------

def _train_fairlearn(data_obj, families, tag_prefix: str = "fl",
                     verbose: bool = True) -> dict:
    """Train fairlearn models and return as {scenario_name: ScenarioModels}.

    Groups models by constraint so each 'scenario' contains LR/RF/XGB results.
    e.g. 'fl_eg_dp' → ScenarioModels with results['lr'], results['rf'], ...
    """
    try:
        from pipeline.fairlearn_pipeline import train_fair_models, _CONSTRAINT_DEFS
    except ImportError:
        warnings.warn("[fairlearn] fairlearn_pipeline not importable — skipping.")
        return {}

    try:
        fl_results = train_fair_models(data_obj, families=families, verbose=verbose)
    except Exception as exc:
        warnings.warn(f"[fairlearn] Training failed: {exc}")
        return {}

    # Group by constraint tag
    by_constraint: dict = {}
    for key, model_res in fl_results.items():
        ctag = getattr(model_res, "constraint_tag", key.rsplit("_", 1)[0])
        fam  = model_res.family
        sc_name = f"{tag_prefix}_{ctag}"
        if sc_name not in by_constraint:
            by_constraint[sc_name] = ml.ScenarioModels(scenario=sc_name)
        by_constraint[sc_name].results[fam] = model_res

    return by_constraint


def _ldp_data_obj(data, X_train_ldp: np.ndarray):
    """Return a lightweight namespace that mimics DataBundle but uses LDP train.

    NOTE: when config.SCALED=True, data.X_train is already scaled, so
    X_train_ldp (which is a copy with the binary race bit possibly flipped)
    is also already on the correct scale. Applying scaler.transform again
    would double-scale and break downstream models.
    """
    return SimpleNamespace(
        X_train    = X_train_ldp,
        y_train    = data.y_train,
        X_train_sc = X_train_ldp,
        X_test     = data.X_test,
        X_test_sc  = data.X_test_sc,
        y_test     = data.y_test,
        scaler     = data.scaler,
        scm        = data.scm,
        feature_cols = data.feature_cols,
    )


# ---------------------------------------------------------------------------
# Attack helper
# ---------------------------------------------------------------------------

def _run_attack(attack_type: str, training_sets: dict, families: list, scenarios: dict,
                data, nice_results: dict, verbose: bool) -> dict:
    try:
        if attack_type == "aia":
            return aia_mod.run_attribute_inference_analysis(
                training_sets=training_sets,
                families=families,
                data=data,
                nice_cf_results=nice_results,
                verbose=verbose,
            )
        return mia_mod.run_mia_analysis(
            scenarios=scenarios, data=data,
            nice_cf_results=nice_results, verbose=verbose)
    except Exception as exc:
        warnings.warn(f"[attack:{attack_type}] Failed: {exc}\n{traceback.format_exc()}")
        return {}


# ===========================================================================
#  MAIN PIPELINE
# ===========================================================================

def run_unified_analysis(
    methods:           list  = ALL_AUG_METHODS,
    families:          list  = MODEL_FAMILIES,
    attack_type:       str   = ATTACKS[1], # "aia",
    skip_mia:          bool  = False,
    skip_nice:         bool  = False,
    ldp_epsilons:      list  = _DEFAULT_EPSILONS,
    include_fairlearn: bool  = True,
    fl_families:       list  = None,       # defaults to ['lr','rf','xgb']
    verbose:           bool  = True,
) -> dict:
    """Run the complete 8-step unified analysis pipeline.

    Returns
    -------
    dict with keys: data, scenarios, fairness, nice_cf, mia, summary
    """
    t0 = time.perf_counter()
    _mkdirs()

    # Fallbacks when the CLI / caller passes None for list arguments.
    methods  = methods  or ALL_AUG_METHODS
    families = families or MODEL_FAMILIES
    fl_fams  = fl_families or ["lr", "rf", "xgb"]
    attack_families = sorted(set(families) | set(fl_fams))
    all_fairness_rows: list  = []
    all_metrics_rows:  list  = []
    all_nice_rows:     list  = []
    all_mia_rows:      list  = []
    all_scenarios:     dict  = {}   # accumulates every trained scenario
    training_sets:     dict  = {}

    # =========================================================================
    # STEP 0  —  Data: load, split, fit SCM
    # =========================================================================
    _banner("STEP 0 — Data Loading & Splitting")
    # Load with the first requested method so we always have data.scm when
    # SCM augmentation is needed; the split is deterministic (fixed seed).
    first_method = methods[0] if methods else "SCM"
    data = dp.load_and_split(augmentation_method=first_method, verbose=verbose)

    prot_test = data.X_test[:, _PROTECTED_IDX]

    def _fairness(scenarios_subset: dict) -> dict:
        """Compute fairness for a subset of scenarios.

        All models are evaluated on the same fixed original test set.
        CF fairness is computed inside compute_fairness by flipping the race
        column on the scaled test set — no external CF array needed.
        """
        results = {}
        for sc_name, sm in scenarios_subset.items():
            results[sc_name] = {}
            for fam, model_res in sm.results.items():
                results[sc_name][fam] = fm.compute_fairness(
                    model_result   = model_res,
                    X_test         = data.X_test_sc,
                    y_test         = data.y_test,
                    protected_test = prot_test,
                    verbose        = verbose,
                )
        return results

    # =========================================================================
    # STEP 1  —  Baseline
    # =========================================================================
    _banner("STEP 1 — Baseline (raw X_train)")
    sc_baseline = ml.train_scenario(
        "baseline", data.X_train_sc, data.y_train,
        data.X_test_sc, data.y_test, families=families, verbose=verbose)
    all_scenarios["baseline"] = sc_baseline
    training_sets["baseline"] = (data.X_train_sc, data.y_train)
    all_metrics_rows += _metrics_rows({"baseline": sc_baseline},
                                       {"block": "fairness", "ldp_eps": None})

    # =========================================================================
    # STEP 2  —  Augmented models (one per method)
    # =========================================================================
    _banner(f"STEP 2 — Augmented training  (methods: {methods})")
    aug_scenarios: dict = {}
    # Snapshot the config flag so we can restore it after the loop.
    import pipeline.config as _cfg
    import pipeline.cf_generation as _cfg_mod
    _saved_bidir = _cfg.AUG_COMPARATORS_BIDIRECTIONAL
    for method in methods:
        sc_name = f"augmented_{method}"
        print(f"\n  [aug] Method: {method}")
        try:
            # Map virtual methods to their underlying augment_data call.
            # "add_comparators_bidir" -> add_comparators with the
            # AUG_COMPARATORS_BIDIRECTIONAL flag forced True for this run.
            if method == "add_comparators_bidir":
                _cfg.AUG_COMPARATORS_BIDIRECTIONAL     = True
                _cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = True
                _real_method = "add_comparators"
            elif method == "add_comparators":
                _cfg.AUG_COMPARATORS_BIDIRECTIONAL     = False
                _cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = False
                _real_method = "add_comparators"
            else:
                _real_method = method
            X_aug, y_aug, _ = augment_data(
                data.X_train, data.y_train, _real_method, data.scm,
                dataset_tag=DATASET_NAME.lower())
            # NOTE: when config.SCALED=True, data.X_train is already scaled,
            # so augment_data returns an already-scaled X_aug. Scaling again
            # would double-scale and crush model performance.
            X_aug_sc = X_aug
            sm = ml.train_scenario(
                sc_name, X_aug_sc, y_aug,
                data.X_test_sc, data.y_test, families=families, verbose=verbose)
            aug_scenarios[sc_name] = sm
            all_scenarios[sc_name] = sm
            training_sets[sc_name] = (X_aug_sc, y_aug)
            all_metrics_rows += _metrics_rows(
                {sc_name: sm},
                {"block": "fairness", "ldp_eps": None, "aug_method": method})
        except Exception as exc:
            warnings.warn(f"[aug] {method} failed: {exc}\n{traceback.format_exc()}")
    # Restore the flag so anything that imports config later sees the
    # user's configured default (not the last per-method override).
    _cfg.AUG_COMPARATORS_BIDIRECTIONAL     = _saved_bidir
    _cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = _saved_bidir

    # =========================================================================
    # STEP 3  —  Fairness: Baseline vs Augmented
    # =========================================================================
    _banner("STEP 3 — Fairness Analysis: Baseline vs Augmented")
    sc_step3 = {"baseline": sc_baseline, **aug_scenarios}
    fair_step3 = _fairness(sc_step3)
    rows_step3 = _fairness_rows(fair_step3, {"block": "baseline_vs_augmented", "ldp_eps": None})
    all_fairness_rows += rows_step3
    _save_csv(pd.DataFrame(rows_step3), "step3_fairness_baseline_vs_augmented.csv")

    if verbose:
        _print_fairness_table(fair_step3)

    # =========================================================================
    # STEP 4  —  Fairlearn baselines
    # =========================================================================
    fl_scenarios: dict = {}
    if include_fairlearn:
        _banner("STEP 4 — Fairlearn Baselines (on raw X_train)")
        fl_scenarios = _train_fairlearn(data, fl_fams, tag_prefix="fl", verbose=verbose)
        for sc_name, sm in fl_scenarios.items():
            all_scenarios[sc_name] = sm
            training_sets[sc_name] = (data.X_train_sc, data.y_train)
            all_metrics_rows += _metrics_rows(
                {sc_name: sm},
                {"block": "fairness", "ldp_eps": None})
    else:
        print("\n[STEP 4] Fairlearn skipped (--no-fairlearn).")

    # =========================================================================
    # STEP 5  —  Full fairness comparison: Baseline + Augmented + Fairlearn
    # =========================================================================
    _banner("STEP 5 — Full Fairness: Baseline + Augmented + Fairlearn")
    sc_step5 = {**sc_step3, **fl_scenarios}
    fair_step5 = _fairness(sc_step5)
    rows_step5 = _fairness_rows(fair_step5, {"block": "all_no_ldp", "ldp_eps": None})
    all_fairness_rows += [r for r in rows_step5
                          if r["scenario"] not in {x["scenario"] for x in rows_step3}]
    _save_csv(pd.DataFrame(rows_step5), "step5_fairness_all_no_ldp.csv")

    if verbose:
        _print_fairness_table(fair_step5)

    # =========================================================================
    # STEP 6  —  LDP: perturb X_train, re-augment, re-train everything
    # =========================================================================
    _banner(f"STEP 6 — LDP (epsilons={ldp_epsilons})")
    ldp_scenarios: dict = {}

    for eps in ldp_epsilons:
        eps_tag = f"eps{eps}"
        print(f"\n  [LDP] epsilon = {eps}")

        # --- 6a: apply LDP to X_train only ---
        X_train_ldp = data.X_train.copy()
        perturbed, p_flip = ldp_mod.randomised_response(
            X_train_ldp[:, _PROTECTED_IDX], epsilon=eps, random_state=RANDOM_STATE)
        X_train_ldp[:, _PROTECTED_IDX] = perturbed
        n_flip = int((perturbed != data.X_train[:, _PROTECTED_IDX]).sum())
        print(f"         p_flip={p_flip:.4f}  flipped={n_flip}/{len(perturbed)}")

        # data.X_train is already scaled (SCALED=True); randomised_response
        # only flips the binary race bit, so X_train_ldp is still on-scale.
        # Do NOT re-apply scaler.transform — that would double-scale.
        X_train_ldp_sc = X_train_ldp

        # --- 6b: LDP-Baseline ---
        sc_name = f"ldp_baseline_{eps_tag}"
        sm = ml.train_scenario(
            sc_name, X_train_ldp_sc, data.y_train,
            data.X_test_sc, data.y_test, families=families, verbose=verbose)
        ldp_scenarios[sc_name] = sm
        all_scenarios[sc_name]  = sm
        training_sets[sc_name] = (X_train_ldp_sc, data.y_train)
        all_metrics_rows += _metrics_rows(
            {sc_name: sm},
            {"block": "ldp", "ldp_eps": eps, "aug_method": "none"})

        # --- 6c: LDP + each augmentation method ---
        # Re-snapshot the bidir flag so we can restore it after this eps loop.
        _saved_bidir_6c = _cfg.AUG_COMPARATORS_BIDIRECTIONAL
        for method in methods:
            sc_name = f"ldp_{method}_{eps_tag}"
            try:
                # Same virtual-method handling as Step 2: route the
                # add_comparators_bidir / add_comparators labels to the real
                # add_comparators implementation, toggling the config flag.
                if method == "add_comparators_bidir":
                    _cfg.AUG_COMPARATORS_BIDIRECTIONAL     = True
                    _cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = True
                    _real_method = "add_comparators"
                elif method == "add_comparators":
                    _cfg.AUG_COMPARATORS_BIDIRECTIONAL     = False
                    _cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = False
                    _real_method = "add_comparators"
                else:
                    _real_method = method
                # Order: LDP-on-X_train FIRST, then augment the LDP'd data.
                # We do NOT apply LDP to the augmented rows themselves.
                X_aug_ldp, y_aug_ldp, _ = augment_data(
                    X_train_ldp, data.y_train, _real_method, data.scm,
                    dataset_tag=DATASET_NAME.lower())  # always "compas" — used as config key
                # X_train_ldp is already scaled, so augment_data returns an
                # already-scaled X_aug_ldp. Re-transforming would double-scale.
                X_aug_ldp_sc = X_aug_ldp
                sm = ml.train_scenario(
                    sc_name, X_aug_ldp_sc, y_aug_ldp,
                    data.X_test_sc, data.y_test, families=families, verbose=verbose)
                ldp_scenarios[sc_name] = sm
                all_scenarios[sc_name]  = sm
                training_sets[sc_name] = (X_aug_ldp_sc, y_aug_ldp)
                all_metrics_rows += _metrics_rows(
                    {sc_name: sm},
                    {"block": "ldp", "ldp_eps": eps, "aug_method": method})
            except Exception as exc:
                warnings.warn(f"[LDP+{method}] eps={eps}: {exc}")
        # Restore bidir flag for this eps iteration so 6d / next eps see clean state.
        _cfg.AUG_COMPARATORS_BIDIRECTIONAL     = _saved_bidir_6c
        _cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = _saved_bidir_6c

        # --- 6d: LDP + Fairlearn ---
        if include_fairlearn:
            ldp_data_obj = _ldp_data_obj(data, X_train_ldp)
            fl_ldp = _train_fairlearn(
                ldp_data_obj, fl_fams,
                tag_prefix=f"fl_ldp_{eps_tag}", verbose=verbose)
            for sc_name, sm in fl_ldp.items():
                ldp_scenarios[sc_name] = sm
                all_scenarios[sc_name]  = sm
                training_sets[sc_name] = (X_train_ldp_sc, data.y_train)
                all_metrics_rows += _metrics_rows(
                    {sc_name: sm},
                    {"block": "ldp", "ldp_eps": eps, "aug_method": "fairlearn"})

    # =========================================================================
    # STEP 7  —  Full privacy + fairness comparison
    # =========================================================================
    _banner("STEP 7 — Full Comparison: All Scenarios")
    fair_step7 = _fairness(all_scenarios)
    rows_step7 = _fairness_rows(fair_step7, {"block": "all_with_ldp"})
    # Only add rows for scenarios not already stored (LDP scenarios are new)
    ldp_sc_names = set(ldp_scenarios.keys())
    all_fairness_rows += [r for r in rows_step7 if r["scenario"] in ldp_sc_names]
    _save_csv(pd.DataFrame(rows_step7), "step7_fairness_all.csv")

    if verbose:
        _print_fairness_table(fair_step7)

    # Save unified model metrics
    _save_csv(pd.DataFrame(all_metrics_rows), "unified_model_metrics.csv")

    # =========================================================================
    # STEP 8  —  AIA + NiCE CFs
    # =========================================================================
    nice_results: dict = {}
    mia_results:  dict = {}

    if not skip_nice:
        _banner(f"STEP 8a — NiCE CF Generation  ({len(all_scenarios)} scenarios)")
        nice_results = ncf.generate_nice_cfs_for_all_models(
            scenarios=all_scenarios,
            data=data,
            max_samples=NICE_MAX_SAMPLES,
            verbose=verbose,
        )
        ncf.save_nice_cf_results(nice_results, _NICE_CACHE_DIR)
        nice_rows = _nice_rows(nice_results)
        all_nice_rows += nice_rows
        _save_csv(pd.DataFrame(nice_rows), "step8_nice_quality.csv")
    else:
        nice_results = ncf.load_nice_cf_results(_NICE_CACHE_DIR)
        if nice_results:
            print(f"\n[STEP 8a] Loaded cached NiCE CFs ({sum(len(v) for v in nice_results.values())} pairs).")
            all_nice_rows += _nice_rows(nice_results)
        else:
            print("\n[STEP 8a] Skipped NiCE CF generation (--skip-nice).")

    if not skip_mia:
        _banner("STEP 8b — {}".format(
            "Attribute Inference Attack" if attack_type == "aia"
            else "Membership Inference Attack"))
        mia_results = _run_attack(attack_type, training_sets, attack_families,
                                  all_scenarios, data, nice_results, verbose)
        if mia_results:
            if attack_type == "aia":
                mia_df = aia_mod.aia_results_to_dataframe(mia_results)
            else:
                mia_df = mia_mod.aia_results_to_dataframe(mia_results)
            _save_csv(mia_df, "step8_aia.csv")
            _save_csv(mia_df, "step8_mia.csv")
            all_mia_rows += mia_df.to_dict("records")
    else:
        print("\n[STEP 8b] Skipped AIA analysis (--skip-aia/--skip-mia).")

    # =========================================================================
    # Build unified comparison summary
    # =========================================================================
    _banner("Building Unified Comparison Summary")

    unified_fairness = pd.DataFrame(all_fairness_rows)
    unified_metrics  = pd.DataFrame(all_metrics_rows)
    unified_nice     = pd.DataFrame(all_nice_rows)
    unified_mia      = pd.DataFrame(all_mia_rows)

    # Save master fairness table (all steps, all scenarios)
    _save_csv(unified_fairness, "unified_fairness.csv")

    # Comparison summary: ONE ROW PER (scenario, family) covering Steps 1-8.
    # Every scenario from every block (baseline, augmented, LDP, fairlearn) is
    # included — earlier versions silently dropped the non-LDP rows.
    summary = _build_summary(unified_fairness, unified_mia, unified_nice, unified_metrics)
    _save_csv(summary, "unified_all_steps_per_scenario.csv")
    # Back-compat alias so older callers still find the file.
    _save_csv(summary, "unified_comparison_summary.csv")

    if not unified_mia.empty:
        _save_csv(unified_mia, "unified_aia.csv")
        _save_csv(unified_mia, "unified_mia.csv")
    if not unified_nice.empty:
        _save_csv(unified_nice, "unified_nice_quality.csv")

    elapsed = time.perf_counter() - t0
    _banner(f"UNIFIED ANALYSIS COMPLETE  ({elapsed:.1f} s)")
    print(f"  Scenarios trained : {len(all_scenarios)}")
    print(f"  Output directory  : {_UNIFIED_RES}")

    return {
        "data":             data,
        "scenarios":        all_scenarios,
        "fairness":         fair_step7,
        "fairness_step3":   fair_step3,
        "fairness_step5":   fair_step5,
        "nice_cf":          nice_results,
        "aia":              mia_results,
        "mia":              mia_results,
        "summary":          summary,
        "unified_fairness": unified_fairness,
        "unified_metrics":  unified_metrics,
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(unified_fairness: pd.DataFrame,
                   unified_mia:      pd.DataFrame,
                   unified_nice:     pd.DataFrame,
                   unified_metrics:  pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, family) combining metrics from every step.

    Steps folded into each row:
      * Step 1 / Step 2 / Step 6 (training)   -> accuracy, auc_roc, f1,
                                                 precision, recall, n_train
                                                 (taken from unified_metrics)
      * Step 3 / Step 4 / Step 5 / Step 7     -> group_* and ind_* fairness
                                                 (taken from unified_fairness —
                                                 every block is unioned so
                                                 no scenario is dropped)
      * Step 8a NiCE CF quality               -> flip_rate, proximity,
                                                 plausibility, sparsity
    * Step 8b AIA on NiCE CFs               -> mia_auc_nice_cf,
                                                 mia_adv_nice_cf (best
                                                 attacker per scenario/family)

    Key context columns kept: scenario, family, block (source block), ldp_eps
    (None for non-LDP), aug_method (none / SCM / update_labels /
    add_comparators / add_comparators_bidir / fairlearn).
    """
    if unified_fairness.empty:
        return pd.DataFrame()

    keys = ["scenario", "family"]

    # --- Fairness: union every block, keep the most complete row per key. ---
    # Earlier versions filtered to block == "all_with_ldp", which dropped all
    # non-LDP scenarios from the summary. Instead, sort so that later blocks
    # ("all_with_ldp" > "all_no_ldp" > "baseline_vs_augmented") win the
    # dedupe — later blocks contain the same fairness numbers plus ldp_eps.
    fair = unified_fairness.copy()
    if "block" in fair.columns:
        _order = {"baseline_vs_augmented": 0, "all_no_ldp": 1, "all_with_ldp": 2}
        fair["_block_rank"] = fair["block"].map(_order).fillna(-1)
        fair = (fair.sort_values(["_block_rank"])
                    .drop_duplicates(subset=keys, keep="last")
                    .drop(columns=["_block_rank"]))
    else:
        fair = fair.drop_duplicates(subset=keys, keep="last")

    fair_cols = keys + [c for c in fair.columns
                        if c.startswith("group_") or c.startswith("ind_")]
    summary = fair[fair_cols].copy()

    # --- Model performance: last row per key wins (LDP overrides non-LDP if
    # the same scenario appears twice, which shouldn't happen because LDP
    # scenarios carry a distinct eps suffix in their name). ---
    if not unified_metrics.empty and all(k in unified_metrics.columns for k in keys):
        perf_cols = [c for c in ["accuracy", "auc_roc", "f1", "precision", "recall",
                                  "n_train", "ldp_eps", "aug_method", "block"]
                     if c in unified_metrics.columns]
        perf = unified_metrics[keys + perf_cols].drop_duplicates(subset=keys, keep="last")
        summary = summary.merge(perf, on=keys, how="left")

    # --- NiCE CF quality (Step 8a) ---
    if not unified_nice.empty and all(k in unified_nice.columns for k in keys):
        nice_cols = [c for c in ["flip_rate", "proximity", "plausibility", "sparsity"]
                     if c in unified_nice.columns]
        if nice_cols:
            nice = unified_nice[keys + nice_cols].drop_duplicates(subset=keys)
            summary = summary.merge(nice, on=keys, how="left")

    # --- MIA on NiCE CFs (Step 8b): pivot EVERY (attacker, metric) into its
    # own column so the row carries the full attack table, plus keep best-
    # attacker convenience columns (mia_best_*). Column naming:
    #   mia_<itype>__<attacker>__<metric>           e.g. mia_nice_cf__lira__auc_roc
    #   mia_best_auc_<itype>, mia_best_adv_<itype>, mia_best_attacker_<itype>
    if not unified_mia.empty and all(k in unified_mia.columns for k in keys):
        if "input_type" in unified_mia.columns:
            # Metrics columns = anything that isn't a key / context label
            _NON_METRIC = set(keys) | {"input_type", "attacker", "block",
                                       "ldp_eps", "aug_method"}
            metric_cols = [c for c in unified_mia.columns if c not in _NON_METRIC]
            for itype in sorted(unified_mia["input_type"].dropna().unique()):
                sub = unified_mia[unified_mia["input_type"] == itype].copy()
                if sub.empty or "attacker" not in sub.columns:
                    continue
                # Wide pivot: one row per (scenario, family); columns are the
                # cross-product of (attacker, metric).
                wide = sub.pivot_table(index=keys, columns="attacker",
                                       values=metric_cols, aggfunc="first")
                # Flatten MultiIndex columns -> "mia_<itype>__<attacker>__<metric>"
                wide.columns = [
                    f"mia_{itype}__{att}__{met}"
                    for met, att in wide.columns.to_flat_index()
                ]
                wide = wide.reset_index()
                summary = summary.merge(wide, on=keys, how="left")
                # Best-attacker convenience columns
                if "auc_roc" in sub.columns:
                    best = (sub.groupby(keys)["auc_roc"].max().reset_index()
                            .rename(columns={"auc_roc": f"mia_best_auc_{itype}"}))
                    summary = summary.merge(best, on=keys, how="left")
                    idx = sub.groupby(keys)["auc_roc"].idxmax()
                    best_att = (sub.loc[idx, keys + ["attacker"]]
                                .rename(columns={"attacker": f"mia_best_attacker_{itype}"}))
                    summary = summary.merge(best_att, on=keys, how="left")
                if "mia_advantage" in sub.columns:
                    adv = (sub.groupby(keys)["mia_advantage"].max().reset_index()
                           .rename(columns={"mia_advantage": f"mia_best_adv_{itype}"}))
                    summary = summary.merge(adv, on=keys, how="left")

    # Reorder columns: keys first, then context, then metric families.
    context = [c for c in ["block", "ldp_eps", "aug_method", "n_train"]
               if c in summary.columns]
    perf_c  = [c for c in ["accuracy", "auc_roc", "f1", "precision", "recall"]
               if c in summary.columns]
    ind_c   = [c for c in summary.columns if c.startswith("ind_")]
    grp_c   = [c for c in summary.columns if c.startswith("group_")]
    nice_c  = [c for c in ["flip_rate", "proximity", "plausibility", "sparsity"]
               if c in summary.columns]
    mia_best = [c for c in summary.columns if c.startswith("mia_best_")]
    mia_wide = [c for c in summary.columns
                if c.startswith("mia_") and c not in mia_best]
    mia_c    = mia_best + mia_wide
    _taken = set(keys + context + perf_c + ind_c + grp_c + nice_c + mia_c)
    leftover = [c for c in summary.columns if c not in _taken]
    summary = summary[keys + context + perf_c + ind_c + grp_c + nice_c + mia_c + leftover]
    return summary


# ---------------------------------------------------------------------------
# Console fairness table
# ---------------------------------------------------------------------------

def _print_fairness_table(fairness: dict) -> None:
    key_metrics = ["SPD", "EOD", "DI", "EqOdds", "AOD", "PP", "cf_fairness"]
    fams = sorted({f for sc in fairness.values() for f in sc.keys()})
    for fam in fams:
        print(f"\n  --- {fam} ---")
        header = ["scenario"] + key_metrics
        print("  " + "  ".join(f"{h:<24}" if i == 0 else f"{h:>12}"
                                for i, h in enumerate(header)))
        for sc_name, sc_fams in fairness.items():
            fr = sc_fams.get(fam)
            if fr is None: continue
            vals = []
            for m in key_metrics:
                v = fr.group.get(m, fr.individual.get(m))
                vals.append("nan" if v is None else f"{v:+.3f}")
            print("  " + "  ".join(
                [f"{sc_name:<24}"] + [f"{v:>12}" for v in vals]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_EPSILONS = LDP_EPSILONS

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Augmentation methods (SCM, update_labels, "
                             "add_comparators, add_comparators_bidir).")
    parser.add_argument("--attack-type", choices=["mia", "aia"], default="aia",
                        help="Choose the attack family for step 8.")
    parser.add_argument("--families", nargs="+",
                        choices=["logistic_regression", "random_forest", "xgboost",
                                 "lr", "rf", "xgb"],
                        default=None,
                        help="Model families (default: all three).")
    parser.add_argument("--ldp-epsilons", nargs="+", type=float,
                        default=_DEFAULT_EPSILONS,
                        help=f"LDP epsilon values (default: {_DEFAULT_EPSILONS}).")
    parser.add_argument("--skip-mia",  action="store_true", help="Skip MIA.")
    parser.add_argument("--skip-aia",  action="store_true", help="Skip AIA.")
    parser.add_argument("--skip-nice", action="store_true", help="Skip NiCE CFs.")
    parser.add_argument("--no-fairlearn", action="store_true",
                        help="Skip fairlearn baselines.")
    parser.add_argument("--fl-families", nargs="+",
                        default=None,
                        help="Fairlearn base families (default: lr rf xgb).")
    parser.add_argument("--quiet", action="store_true", help="Less output.")
    args = parser.parse_args()

    methods = args.methods or ALL_AUG_METHODS
    # Normalize family short names (lr/rf/xgb) to canonical model names.
    _FAM_ALIAS = {"lr": "logistic_regression", "rf": "random_forest", "xgb": "xgboost"}
    families_norm = ([_FAM_ALIAS.get(f, f) for f in args.families]
                     if args.families else None)
    run_unified_analysis(
        methods            = methods,
        families           = families_norm,
        attack_type        = args.attack_type,
        ldp_epsilons       = args.ldp_epsilons,
        skip_mia           = (args.skip_aia or args.skip_mia),
        skip_nice          = args.skip_nice,
        include_fairlearn  = not args.no_fairlearn,
        fl_families        = args.fl_families,
        verbose            = not args.quiet,
    )
