from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_BASE_DIR, os.path.join(_BASE_DIR, "ensemble_mia-main")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.config import FEATURE_COLS, PROTECTED_COL, RANDOM_STATE, OUTPUT_DIR, SCALED
from pipeline.data_preparation import augment_data, load_and_split
from pipeline.models import _build_estimator


@dataclass
class AttackResult:
    attacker_name: str
    model_name: str
    input_type: str
    metrics: Dict[str, float] = field(default_factory=dict)
    fpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class AIAResult:
    model_name: str
    input_type: str
    attacks: Dict[str, AttackResult] = field(default_factory=dict)
    score_df: Optional[pd.DataFrame] = None
    true_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    runtime_sec: float = 0.0


def _compute_metrics(true_labels, scores):
    true_labels = np.asarray(true_labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(true_labels):
        return {}, np.array([]), np.array([])
    if np.isnan(scores).all() or np.unique(true_labels).size < 2:
        return {}, np.array([]), np.array([])

    threshold = np.median(scores)
    y_pred = (scores >= threshold).astype(int)
    tp = int(((true_labels == 1) & (y_pred == 1)).sum())
    fp = int(((true_labels == 0) & (y_pred == 1)).sum())
    tn = int(((true_labels == 0) & (y_pred == 0)).sum())
    fn = int(((true_labels == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fpr_curve, tpr_curve, _ = roc_curve(true_labels, scores)
    tpr_at = {f"tpr_at_fpr_{t}": float(np.interp(t, fpr_curve, tpr_curve)) for t in [0.0, 0.001, 0.01, 0.1]}

    _advantage = float(tpr - fpr)
    _privacy_gain = float(1.0 - _advantage)
    metrics = {
        "auc_roc": float(roc_auc_score(true_labels, scores)),
        "pr_auc": float(average_precision_score(true_labels, scores)),
        "accuracy": float(accuracy_score(true_labels, y_pred)),
        "precision": float(precision_score(true_labels, y_pred, zero_division=0)),
        "recall": float(recall_score(true_labels, y_pred, zero_division=0)),
        "f1": float(f1_score(true_labels, y_pred, zero_division=0)),
        "true_positive_rate": float(tpr),
        "false_positive_rate": float(fpr),
        "aia_advantage": _advantage,
        "mia_advantage": _advantage,           # alias for visualization compatibility
        "attribute_privacy_gain": _privacy_gain,
        "privacy_gain": _privacy_gain,         # alias for visualization compatibility
        **tpr_at,
    }
    return metrics, fpr_curve, tpr_curve


def _infer_sensitive_binary(values, low_val: float, high_val: float):
    threshold = (float(low_val) + float(high_val)) / 2.0
    return (np.asarray(values, dtype=float) >= threshold).astype(int)


def _drop_sensitive_column(X, sensitive_idx: int):
    return np.delete(np.asarray(X, dtype=float), sensitive_idx, axis=1)


def _build_attack_features(estimator, X, sensitive_idx: int):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0)), np.array([]), np.array([]), np.array([])

    X_ns = _drop_sensitive_column(X, sensitive_idx)
    proba = estimator.predict_proba(X_ns)[:, 1]
    pred = estimator.predict(X_ns).astype(float)
    confidence = np.maximum(proba, 1.0 - proba)
    # Standard AIA setup: attacker sees only target-model outputs, not the raw row.
    features = np.column_stack([proba, pred, confidence])
    return features, proba, pred, confidence


def _build_ensemble_score(score_map: Dict[str, np.ndarray], eval_size: int):
    """Combine attack scores into one ensemble score.

    Prefer actual attacker outputs when they exist. Fallback to probability-type
    baseline scores only when no attacker model could be trained. target_pred is
    always excluded because it is a hard binary label (0/1), not a probability,
    and averaging it with probability scores corrupts the ensemble signal.
    """
    attack_keys = [
        key for key in ("attacker_lr", "attacker_rf")
        if key in score_map and len(score_map[key]) == eval_size
    ]
    if attack_keys:
        return np.mean(np.column_stack([score_map[key] for key in attack_keys]), axis=1)

    # Fallback: use only probability-type scores (exclude hard binary target_pred)
    prob_keys = ("target_prob_pos", "target_confidence")
    valid_scores = [
        score_map[k] for k in prob_keys
        if k in score_map and len(score_map[k]) == eval_size
    ]
    if valid_scores:
        return np.mean(np.column_stack(valid_scores), axis=1)
    return None


def _train_target_estimator(family: str, X_train, y_train, sensitive_idx: int):
    estimator = _build_estimator(family)
    estimator.fit(_drop_sensitive_column(X_train, sensitive_idx), y_train)
    return estimator


def run_attribute_inference_single(
    model_name,
    input_type,
    estimator,
    X_train,
    X_eval,
    sensitive_idx: int,
    low_val: float,
    high_val: float,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
):
    t0 = time.perf_counter()
    X_train = np.asarray(X_train, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    result = AIAResult(model_name=model_name, input_type=input_type)

    if X_train.size == 0 or X_eval.size == 0:
        result.runtime_sec = time.perf_counter() - t0
        return result

    y_train_sensitive = _infer_sensitive_binary(X_train[:, sensitive_idx], low_val, high_val)
    y_eval_sensitive = _infer_sensitive_binary(X_eval[:, sensitive_idx], low_val, high_val)
    result.true_labels = y_eval_sensitive

    F_train, prob_train, pred_train, conf_train = _build_attack_features(
        estimator, X_train, sensitive_idx
    )
    F_eval, prob_eval, pred_eval, conf_eval = _build_attack_features(
        estimator, X_eval, sensitive_idx
    )

    if len(y_eval_sensitive) == 0 or np.unique(y_eval_sensitive).size < 2:
        result.runtime_sec = time.perf_counter() - t0
        return result

    score_map: Dict[str, np.ndarray] = {
        "target_prob_pos": prob_eval,
        "target_pred": pred_eval,
        "target_confidence": conf_eval,
    }

    if np.unique(y_train_sensitive).size >= 2 and F_train.shape[0] > 1 and F_eval.shape[0] > 0:
        try:
            atk_lr = LogisticRegression(max_iter=400, random_state=random_state)
            atk_lr.fit(F_train, y_train_sensitive)
            score_map["attacker_lr"] = atk_lr.predict_proba(F_eval)[:, 1]
        except Exception as exc:
            warnings.warn(f"[AIA] attacker_lr failed: {exc}")

        try:
            atk_rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=random_state,
            )
            atk_rf.fit(F_train, y_train_sensitive)
            score_map["attacker_rf"] = atk_rf.predict_proba(F_eval)[:, 1]
        except Exception as exc:
            warnings.warn(f"[AIA] attacker_rf failed: {exc}")

    ensemble_score = _build_ensemble_score(score_map, len(y_eval_sensitive))
    if ensemble_score is not None:
        score_map["ensemble_mean"] = ensemble_score

    for att_name, score in score_map.items():
        try:
            metrics, fpr_curve, tpr_curve = _compute_metrics(y_eval_sensitive, score)
        except Exception as exc:
            warnings.warn(f"[AIA] metrics for {att_name} failed: {exc}")
            continue
        if not metrics:
            continue
        result.attacks[att_name] = AttackResult(
            attacker_name=att_name,
            model_name=model_name,
            input_type=input_type,
            metrics=metrics,
            fpr_curve=fpr_curve,
            tpr_curve=tpr_curve,
        )

    result.runtime_sec = time.perf_counter() - t0
    if verbose and result.attacks:
        best = max(result.attacks.values(), key=lambda r: r.metrics.get("auc_roc", 0.0))
        print(
            f"  [AIA] Best: {best.attacker_name}  AUC={best.metrics.get('auc_roc', float('nan')):.3f}  "
            f"Adv={best.metrics.get('aia_advantage', float('nan')):+.3f}  ({result.runtime_sec:.1f}s)"
        )
    return result


def _resolve_training_pair(training_sets: dict, scenario_name: str):
    if scenario_name not in training_sets:
        raise KeyError(f"Missing training set for scenario {scenario_name!r}")
    pair = training_sets[scenario_name]
    if isinstance(pair, dict):
        return pair["X_train"], pair["y_train"]
    return pair


def _unpack_pair(pair):
    if isinstance(pair, dict):
        if "X_train" in pair and "y_train" in pair:
            return pair["X_train"], pair["y_train"]
        if "X_eval" in pair and "y_eval" in pair:
            return pair["X_eval"], pair["y_eval"]
        raise KeyError("Expected a pair with X_train/y_train or X_eval/y_eval")
    return pair


def run_attribute_inference_analysis(
    training_sets: dict,
    families,
    data,
    nice_cf_results: Optional[dict] = None,
    sensitive_col: str = PROTECTED_COL,
    verbose: bool = True,
):
    """Attribute inference on target models trained without the sensitive feature.

    Parameters
    ----------
    training_sets:
        Mapping scenario name -> (X_train, y_train) used to retrain the target
        model without the protected column.
    families:
        Model families to train for the target model and the attack model.
    data:
        DataBundle or compatible object providing X_test_sc, scaler and the
        protected feature layout.
    nice_cf_results:
        Optional NiCE counterfactual results for the same scenario/family keys.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  ATTRIBUTE INFERENCE ATTACK  (target model excludes sensitive feature)")
        print("=" * 60)

    if sensitive_col not in FEATURE_COLS:
        warnings.warn(f"[AIA] Unknown sensitive_col={sensitive_col!r}; using {PROTECTED_COL!r}.")
        sensitive_col = PROTECTED_COL
    sensitive_idx = FEATURE_COLS.index(sensitive_col)

    X_train_ref = np.asarray(next(iter(training_sets.values()))[0], dtype=float)
    col_train = X_train_ref[:, sensitive_idx]
    low_val = float(np.nanmin(col_train))
    high_val = float(np.nanmax(col_train))
    if not np.isfinite(low_val) or not np.isfinite(high_val) or low_val == high_val:
        low_val, high_val = 0.0, 1.0

    attack_source_X = np.asarray(data.X_train_sc, dtype=float)
    attack_source_y = np.asarray(data.y_train, dtype=int)
    if attack_source_X.shape[0] != attack_source_y.shape[0]:
        raise ValueError("AIA attack source rows and labels must have the same length")

    attack_strat = np.array([
        f"{label}_{sensitive}"
        for label, sensitive in zip(
            attack_source_y.astype(int),
            attack_source_X[:, sensitive_idx].astype(int),
        )
    ])
    attack_X_train, attack_X_eval, attack_y_train, attack_y_eval = train_test_split(
        attack_source_X,
        attack_source_y,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=attack_strat,
    )

    all_results = {}
    for sc_name in training_sets.keys():
        X_train_sc, y_train_sc = _resolve_training_pair(training_sets, sc_name)
        X_train_sc = np.asarray(X_train_sc, dtype=float)
        y_train_sc = np.asarray(y_train_sc, dtype=int)
        all_results[sc_name] = {}

        for family in families:
            all_results[sc_name][family] = {}
            if verbose:
                print(f"\n[AIA] Scenario={sc_name}  Family={family}  attr={sensitive_col}")

            target_estimator = _train_target_estimator(family, X_train_sc, y_train_sc, sensitive_idx)
            model_name = f"{sc_name}_{family}_nosens"

            all_results[sc_name][family]["test_rows"] = run_attribute_inference_single(
                model_name=model_name,
                input_type="orig_train_eval",
                estimator=target_estimator,
                X_train=attack_X_train,
                X_eval=attack_X_eval,
                sensitive_idx=sensitive_idx,
                low_val=low_val,
                high_val=high_val,
                verbose=verbose,
            )

            # if nice_cf_results:
            #     nice_res = nice_cf_results.get(sc_name, {}).get(family)
            # else:
            #     nice_res = None

            # if nice_res is not None and getattr(nice_res, "X_cf", None) is not None and len(nice_res.X_cf) > 0:
            #     X_eval_nice = data.scaler.transform(nice_res.X_cf) if SCALED else np.asarray(nice_res.X_cf, dtype=float)
            #     all_results[sc_name][family]["nice_cf"] = run_attribute_inference_single(
            #         model_name=model_name,
            #         input_type="nice_cf",
            #         estimator=target_estimator,
            #         X_train=X_train_sc,
            #         X_eval=X_eval_nice,
            #         sensitive_idx=sensitive_idx,
            #         low_val=low_val,
            #         high_val=high_val,
            #         verbose=verbose,
            #     )
            # else:
            #    all_results[sc_name][family]["nice_cf"] = AIAResult(model_name, "nice_cf")
                # if verbose and nice_cf_results is not None:
            # NiCE evaluation is intentionally disabled in this AIA path.

    return all_results


def run_aia_analysis(*args, **kwargs):
    """Back-compat wrapper: legacy name now runs attribute inference."""
    return run_attribute_inference_analysis(*args, **kwargs)


def aia_results_to_dataframe(all_results):
    rows = []
    for sc, families in all_results.items():
        for fam, inputs in families.items():
            for itype, aia_res in inputs.items():
                for att_name, att_res in aia_res.attacks.items():
                    row = {"scenario": sc, "family": fam, "input_type": itype, "attacker": att_name}
                    row.update(att_res.metrics)
                    rows.append(row)
    return pd.DataFrame(rows)


def run_standalone_attack(
    methods=None,
    families=None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
):
    from pipeline.config import MODEL_FAMILIES, AUGMENTATION_METHOD

    methods = methods or [AUGMENTATION_METHOD]
    families = families or MODEL_FAMILIES
    out = output_dir or OUTPUT_DIR
    data = load_and_split(augmentation_method=methods[0], output_dir=out, verbose=verbose)

    training_sets = {"baseline": (data.X_train_sc, data.y_train)}
    for method in methods:
        X_aug, y_aug, _ = augment_data(data.X_train, data.y_train, method, data.scm, dataset_tag="compas", verbose=verbose)
        training_sets[f"augmented_{method}"] = (X_aug, y_aug)

    results = run_attribute_inference_analysis(
        training_sets=training_sets,
        families=families,
        data=data,
        nice_cf_results={},
        verbose=verbose,
    )
    df = aia_results_to_dataframe(results)
    path = os.path.join(out, "results", "attribute_inference_results.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        print(f"[AIA] Saved -> {path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the attribute inference attack independently.")
    parser.add_argument("--methods", nargs="+", default=None, help="Augmentation methods to include (default: current config method).")
    parser.add_argument("--families", nargs="+", choices=["logistic_regression", "random_forest", "xgboost", "lr", "rf", "xgb"], default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    alias = {"lr": "logistic_regression", "rf": "random_forest", "xgb": "xgboost"}
    families = [alias.get(n, n) for n in (args.families or [])] or None
    run_standalone_attack(methods=args.methods, families=families, verbose=not args.quiet)