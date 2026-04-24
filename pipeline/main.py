"""
main.py -- Full orchestration of the COMPAS fairness / privacy pipeline.

Scenarios
---------
  baseline              : LR/RF/XGB trained on original training data
  augmented             : LR/RF/XGB trained on original + MM race CFs
  ldp_baseline_eps{e}   : LR/RF/XGB trained on LDP(race_enc, eps=e) only
                          (no MM-CF augmentation -- pure privacy scenario)
  ldp_augmented_eps{e}  : LR/RF/XGB trained on LDP(race_enc, eps=e) + MM race CFs

  Default epsilons come from config.LDP_EPSILONS (currently [0.5, 1.0, 5.0]).

Analyses
--------
  1. Fairness  -- SPD, DI, EOD, AOD, EqOdds, PP, Theil,
                  CF-Fairness (SCM race CFs), k-NN Consistency
                  Reported for ALL scenarios and ALL model families.

  2. NiCE CFs  -- Generated for ALL scenarios.
                  Quality metrics: flip_rate, proximity, plausibility, sparsity.

  3. MIA       -- NiCE CFs used as synth input for ALL scenarios.
                  Metrics: AUC-ROC, MIA Advantage, Privacy Gain,
                           TPR@FPR={0, 0.001, 0.01, 0.1}.

  4. LDP sweep -- Privacy-fairness trade-off across epsilons saved as
                  ldp_sweep_results.csv.

Usage
-----
  python pipeline/main.py                              # full run
  python pipeline/main.py --skip-mia                  # skip MIA
  python pipeline/main.py --skip-nice                 # skip NiCE
  python pipeline/main.py --ldp-epsilons 0.5 1.0 5.0  # custom epsilons
  python pipeline/main.py --families lr rf            # subset of models
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings

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
    OUTPUT_DIR, MODEL_FAMILIES, LDP_EPSILONS, NICE_MAX_SAMPLES, AUGMENTATION_METHOD,
)
import pipeline.data_preparation as dp
import pipeline.models            as ml
import pipeline.fairness_metrics  as fm
import pipeline.nice_cf           as ncf
import pipeline.ldp               as ldp_mod
import pipeline.mia_analysis      as mia_mod
import pipeline.visualizations    as viz

# Default LDP epsilon sweep -- edit LDP_EPSILONS in config.py
_DEFAULT_LDP_EPSILONS = LDP_EPSILONS


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _results_dir(out: str = None) -> str:
    path = os.path.join(out or OUTPUT_DIR, "results")
    os.makedirs(path, exist_ok=True)
    return path


def _banner(text: str) -> None:
    b = "=" * 60
    print("\n{}\n  {}\n{}".format(b, text, b))


def _save_fairness_csv(fairness_results: dict, out: str = None) -> str:
    """Flatten ALL scenarios x families x metrics into one CSV."""
    rows = []
    for sc, families in fairness_results.items():
        for fam, fr in families.items():
            row = {"scenario": sc, "family": fam, "model": fr.model_name}
            row.update({"group_" + k: v for k, v in fr.group.items()})
            row.update({"ind_"   + k: v for k, v in fr.individual.items()})
            rows.append(row)
    df   = pd.DataFrame(rows)
    path = os.path.join(_results_dir(out), "fairness_results.csv")
    df.to_csv(path, index=False)
    print("  [results] Fairness CSV ({} rows) -> {}".format(len(df), path))
    return path


def _save_mia_csv(mia_df: pd.DataFrame, out: str = None) -> str:
    path = os.path.join(_results_dir(out), "mia_results.csv")
    mia_df.to_csv(path, index=False)
    print("  [results] MIA CSV ({} rows) -> {}".format(len(mia_df), path))
    return path


def _save_model_metrics_csv(scenarios: dict, out: str = None) -> str:
    """Save accuracy/precision/recall/F1/AUC for every scenario × family."""
    rows = []
    for sc_name, sm in scenarios.items():
        for fam, res in sm.results.items():
            m = res.metrics
            if not m:
                continue
            rows.append({
                "scenario":  sc_name,
                "family":    fam,
                "model":     res.name,
                "accuracy":  m.get("accuracy",  float("nan")),
                "auc_roc":   m.get("auc_roc",   float("nan")),
                "f1":        m.get("f1",         float("nan")),
                "precision": m.get("precision",  float("nan")),
                "recall":    m.get("recall",     float("nan")),
            })
    df   = pd.DataFrame(rows)
    path = os.path.join(_results_dir(out), "model_metrics.csv")
    df.to_csv(path, index=False)
    print("  [results] Model metrics CSV ({} rows) -> {}".format(len(df), path))
    return path


def _save_nice_quality_csv(nice_results: dict, out: str = None) -> str:
    """Save NiCE CF quality including flip_rate."""
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
            rows.append(row)
    df   = pd.DataFrame(rows)
    path = os.path.join(_results_dir(out), "nice_cf_quality.csv")
    df.to_csv(path, index=False)
    print("  [results] NiCE CF quality CSV ({} rows) -> {}".format(len(df), path))
    return path


def _save_ldp_sweep_csv(ldp_results: dict, fairness_results: dict,
                         mia_results: dict, families: list,
                         scenarios: dict = None, out: str = None) -> str:
    """Tidy CSV summarising the privacy-fairness trade-off across epsilons.

    Covers both ldp_baseline_eps{e} (pure privacy) and
    ldp_augmented_eps{e} (privacy + MM-CF augmentation).
    """
    rows = []
    for ldp_label, lr in ldp_results.items():
        eps = lr.epsilon
        # Build the two scenario labels derived from this LDP result
        sc_labels = [
            (ldp_label.replace("ldp_eps", "ldp_baseline_eps"),  "baseline+LDP"),
            (ldp_label.replace("ldp_eps", "ldp_augmented_eps"), "augmented+LDP"),
        ]
        for sc_label, sc_type in sc_labels:
            for fam in families:
                row = {
                    "scenario":   sc_label,
                    "sc_type":    sc_type,
                    "epsilon":    eps,
                    "flip_prob":  lr.flip_prob,
                    "n_flipped":  lr.n_flipped,
                    "family":     fam,
                }
                # Fairness
                fr = fairness_results.get(sc_label, {}).get(fam)
                if fr is not None:
                    row.update({"group_" + k: v for k, v in fr.group.items()})
                    row.update({"ind_"   + k: v for k, v in fr.individual.items()})
                # MIA -- nice_cf best attacker
                mia_fam  = mia_results.get(sc_label, {}).get(fam, {})
                mia_nice = mia_fam.get("nice_cf")
                if mia_nice and getattr(mia_nice, "attacks", {}):
                    best = max(mia_nice.attacks.values(),
                               key=lambda r: r.metrics.get("auc_roc", 0))
                    row["mia_best_attacker"] = best.attacker_name
                    row["mia_auc_roc"]       = best.metrics.get("auc_roc",       float("nan"))
                    row["mia_advantage"]     = best.metrics.get("mia_advantage", float("nan"))
                    row["mia_privacy_gain"]  = best.metrics.get("privacy_gain",  float("nan"))
                rows.append(row)
    df   = pd.DataFrame(rows)
    path = os.path.join(_results_dir(out), "ldp_sweep_results.csv")
    df.to_csv(path, index=False)
    print("  [results] LDP sweep CSV ({} rows) -> {}".format(len(df), path))
    return path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(fairness_results: dict, mia_results: dict) -> None:
    """Print a concise comparison table for all scenarios."""
    if not fairness_results:
        return
    key_metrics = ["SPD", "EOD", "DI", "EqOdds", "AOD", "PP", "cf_fairness"]
    print("\n  Fairness across all scenarios (SPD / EOD / DI / EqOdds / AOD / PP / CF-Fair):")
    hdr = "  {:<20}  {:<22}".format("Scenario", "Family") + "".join(
        "  {:>9}".format(k) for k in key_metrics)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for sc, families in sorted(fairness_results.items()):
        for fam, fr in families.items():
            row = "  {:<20}  {:<22}".format(sc, fam)
            for k in key_metrics:
                v = fr.group.get(k, fr.individual.get(k, float("nan")))
                if isinstance(v, float) and v == v:
                    row += "  {:>+9.4f}".format(v)
                else:
                    row += "  {:>9}".format("N/A")
            print(row)

    if mia_results:
        try:
            mia_df = mia_mod.mia_results_to_dataframe(mia_results)
            if not mia_df.empty and "auc_roc" in mia_df.columns:
                best = (mia_df.groupby(["scenario", "family", "input_type"])
                        ["auc_roc"].max().reset_index())
                for itype in ["mm_cf", "nice_cf"]:
                    sub = best[best["input_type"] == itype]
                    if not sub.empty:
                        pivot = sub.pivot(index="scenario", columns="family",
                                          values="auc_roc")
                        print("\n  MIA best-attacker AUC-ROC ({} proxy):".format(itype))
                        print("  " + pivot.to_string())
        except Exception as exc:
            print("  [MIA summary] {}".format(exc))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    families:             list = MODEL_FAMILIES,
    skip_mia:             bool = False,
    skip_nice:            bool = False,
    ldp_epsilons:         list = _DEFAULT_LDP_EPSILONS,
    augmentation_method:  str  = AUGMENTATION_METHOD,
    output_dir:           str  = None,
    verbose:              bool = True,
) -> dict:
    """Execute the full multi-scenario fairness / privacy pipeline.

    Parameters
    ----------
    augmentation_method : 'SCM' | 'update_labels' | 'add_comparators'
        Controls how the augmented training set is built.
    output_dir : override for OUTPUT_DIR (used when sweeping multiple methods)
    """
    t_start = time.perf_counter()
    _out = output_dir or OUTPUT_DIR
    os.makedirs(_out, exist_ok=True)
    results: dict = {}

    # ── STEP 0: Data ──────────────────────────────────────────────────
    _banner("STEP 0 -- Data Loading & Splitting  [aug={}]".format(augmentation_method))
    data = dp.load_and_split(augmentation_method=augmentation_method,
                             output_dir=_out, verbose=verbose)
    results["data"] = data

    # ── STEP 1: LDP for each epsilon ──────────────────────────────────
    _banner("STEP 1 -- Local Differential Privacy  (epsilons={})".format(
        ldp_epsilons))
    ldp_results = ldp_mod.apply_ldp_multi_eps(
        data, epsilons=ldp_epsilons, verbose=verbose)
    results["ldp"] = ldp_results

    # Print p_flip table
    if verbose:
        print("\n  Epsilon -> flip_prob mapping:")
        print("  {:>8}  {:>10}  {:>10}".format("epsilon", "p_flip", "n_flipped"))
        print("  " + "-" * 32)
        for lbl, lr in ldp_results.items():
            print("  {:>8.2f}  {:>10.4f}  {:>10,}".format(
                lr.epsilon, lr.flip_prob, lr.n_flipped))

    # ── STEP 2: Model training ─────────────────────────────────────────
    _banner("STEP 2 -- Model Training (all scenarios)")
    scenarios: dict = {}

    if verbose:
        print("\n[Scenario 1] baseline -- original training data")
    scenarios["baseline"] = ml.train_scenario(
        "baseline", data.X_train_sc, data.y_train,
        data.X_test_sc, data.y_test, families=families, verbose=verbose)

    aug_label = "augmented_{}".format(augmentation_method)
    if verbose:
        print("\n[Scenario 2] {} -- original + {} augmentation".format(
            aug_label, augmentation_method))
    scenarios[aug_label] = ml.train_scenario(
        aug_label, data.X_aug_sc, data.y_aug,
        data.X_test_sc, data.y_test, families=families, verbose=verbose)

    for label, lr in ldp_results.items():
        # Scenario 4: LDP on baseline (pure privacy, no MM-CFs)
        bl_label = label.replace("ldp_eps", "ldp_baseline_eps")
        if verbose:
            print("\n[Scenario 4] {} -- LDP(eps={}) on original data only".format(
                bl_label, lr.epsilon))
        scenarios[bl_label] = ml.train_scenario(
            bl_label, lr.X_train_ldp_sc, lr.y_train_ldp,
            data.X_test_sc, data.y_test, families=families, verbose=verbose)

        # Scenario 5b: LDP on augmented (privacy + MM-CF fairness)
        aug_label = label.replace("ldp_eps", "ldp_augmented_eps")
        if verbose:
            print("\n[Scenario 5b] {} -- LDP(eps={}) + MM-CFs".format(
                aug_label, lr.epsilon))
        scenarios[aug_label] = ml.train_scenario(
            aug_label, lr.X_aug_ldp_sc, lr.y_aug_ldp,
            data.X_test_sc, data.y_test, families=families, verbose=verbose)

    results["scenarios"] = scenarios

    # ── STEP 3: NiCE CFs (ALL scenarios) ──────────────────────────────
    _nice_cf_dir = os.path.join(_out, "results", "nice_cf_arrays")
    nice_cf_results: dict = {}
    if not skip_nice:
        _banner("STEP 3 -- NiCE CF Generation  (all {} scenarios, "
                "max_samples={})".format(len(scenarios), NICE_MAX_SAMPLES))
        nice_cf_results = ncf.generate_nice_cfs_for_all_models(
            scenarios=scenarios,
            data=data,
            max_samples=NICE_MAX_SAMPLES,
            verbose=verbose,
        )
        ncf.save_nice_cf_results(nice_cf_results, _nice_cf_dir)
    else:
        nice_cf_results = ncf.load_nice_cf_results(_nice_cf_dir)
        if nice_cf_results:
            print("\n[STEP 3] Loaded cached NiCE CFs from disk ({} scenario/family pairs).".format(
                sum(len(v) for v in nice_cf_results.values())))
        else:
            print("\n[STEP 3] Skipped NiCE CF generation (--skip-nice). No cached CFs found.")
    results["nice_cf"] = nice_cf_results

    # ── STEP 4: Fairness metrics (ALL scenarios) ───────────────────────
    _banner("STEP 4 -- Fairness Metrics (all {} scenarios)".format(len(scenarios)))
    fairness_results = fm.run_fairness_analysis(
        scenarios=scenarios, data=data, verbose=verbose)
    results["fairness"] = fairness_results
    _save_fairness_csv(fairness_results, out=_out)
    _save_model_metrics_csv(scenarios, out=_out)

    # ── STEP 5: MIA (ALL scenarios, two proxies) ───────────────────────
    mia_results: dict = {}
    if not skip_mia:
        _banner("STEP 5 -- Membership Inference Attack (all scenarios)")
        try:
            mia_results = mia_mod.run_mia_analysis(
                scenarios=scenarios,
                data=data,
                nice_cf_results=nice_cf_results,
                verbose=verbose,
            )
            mia_df = mia_mod.mia_results_to_dataframe(mia_results)
            _save_mia_csv(mia_df, out=_out)
            results["mia"] = mia_results
        except Exception as e:
            warnings.warn("MIA error: {}\n{}".format(e, traceback.format_exc()))
            results["mia"] = {}
    else:
        print("\n[STEP 5] Skipped MIA analysis (--skip-mia).")
        results["mia"] = {}

    # ── STEP 6: LDP sweep summary ──────────────────────────────────────
    _banner("STEP 6 -- LDP Sweep Summary & Unified Comparison")
    _print_summary(fairness_results, results.get("mia", {}))
    _save_ldp_sweep_csv(ldp_results, fairness_results,
                        results.get("mia", {}), families,
                        scenarios=scenarios, out=_out)

    # ── STEP 7: NiCE quality CSV ───────────────────────────────────────
    if nice_cf_results:
        _save_nice_quality_csv(nice_cf_results, out=_out)

    # ── STEP 8: Figures ────────────────────────────────────────────────
    _banner("STEP 8 -- Generating Figures")
    try:
        fig_paths = viz.generate_all_figures(
            fairness_results=fairness_results,
            all_mia_results=results.get("mia", {}),
            nice_cf_results=nice_cf_results,
            verbose=verbose,
        )
        results["figures"] = fig_paths
    except Exception as e:
        warnings.warn("Figure generation failed: {}\n{}".format(
            e, traceback.format_exc()))
        results["figures"] = []

    elapsed = time.perf_counter() - t_start
    _banner("PIPELINE COMPLETE  [aug={}]  ({:.1f} s)".format(
        augmentation_method, elapsed))
    print("\n  Output directory : {}".format(_out))
    print("  Results CSVs     : {}".format(_results_dir(_out)))
    print("  Scenarios run    : {}".format(list(scenarios.keys())))
    print("  Figures saved    : {}".format(len(results.get("figures", []))))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COMPAS Fairness / Privacy Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-mia",  action="store_true",
                        help="Skip MIA analysis.")
    parser.add_argument("--skip-nice", action="store_true",
                        help="Skip NiCE CF generation.")
    parser.add_argument("--ldp-epsilons", nargs="+", type=float,
                        default=_DEFAULT_LDP_EPSILONS,
                        help=("LDP epsilon values to sweep. "
                              "Default: {} (edit LDP_EPSILONS in config.py)".format(
                                  _DEFAULT_LDP_EPSILONS)))
    parser.add_argument("--families", nargs="+",
                        choices=["logistic_regression", "random_forest",
                                 "xgboost", "lr", "rf", "xgb"],
                        default=None, help="Model families to train.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output.")
    args = parser.parse_args()

    alias = {"lr": "logistic_regression", "rf": "random_forest",
             "xgb": "xgboost"}
    families = [alias.get(n, n) for n in (args.families or MODEL_FAMILIES)]
    run_pipeline(
        families=families,
        skip_mia=args.skip_mia,
        skip_nice=args.skip_nice,
        ldp_epsilons=args.ldp_epsilons,
        verbose=not args.quiet,
    )
