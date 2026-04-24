"""
Base class for all Membership Inference Attack (MIA) methods.

Every attacker must:
  - Inherit from BaseAttacker
  - Override _compute_attack_scores(X_test, synth, ref) → np.ndarray of scores

Convention (shared across all subclasses except local_neighborhood):
    _compute_attack_scores(X_test, synth, ref)
        X_test : np.ndarray, shape (N, d) — member + non-member pool
        synth  : np.ndarray, shape (M, d) — proxy/synthetic/CF data
        ref    : np.ndarray, shape (R, d) — population reference data (may be None)
    Returns scores where *higher* = more likely to be a member.

The public attack() method builds X_test and labels from (mem, non_mem) and
delegates to _compute_attack_scores.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .evaluation import AttackEvaluator


class BaseAttacker(ABC):
    """Abstract base class for all MIA attackers."""

    def __init__(self, hyper_parameters: Optional[Dict[str, Any]] = None) -> None:
        self.hyper_parameters: Dict[str, Any] = hyper_parameters or {}
        self.name: str = "BaseAttacker"

    # ------------------------------------------------------------------
    # Abstract method — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_attack_scores(
        self,
        X_test: np.ndarray,
        synth: np.ndarray,
        ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute a membership score for every row in X_test.

        Args:
            X_test: Rows whose membership we want to infer (members first,
                    then non-members, as stacked by attack()).
            synth:  Proxy/synthetic/CF dataset available to the adversary.
            ref:    Population reference dataset (optional, may be None).

        Returns:
            1-D array of length len(X_test).  Higher score ↔ more likely member.
        """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def attack(
        self,
        mem: np.ndarray,
        non_mem: np.ndarray,
        synth: np.ndarray,
        ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the attack and return (true_labels, predicted_scores).

        Args:
            mem:     Training members to test  (label = 1).
            non_mem: Non-members to test        (label = 0).
            synth:   Proxy data the adversary has access to (CFs, synthetic data …).
            ref:     Population reference data (optional).

        Returns:
            labels: np.ndarray of shape (len(mem)+len(non_mem),)  — 1=member, 0=non-member
            scores: np.ndarray of same shape — continuous membership scores
        """
        X_test = np.vstack([mem, non_mem])
        labels = np.concatenate(
            [np.ones(len(mem), dtype=int), np.zeros(len(non_mem), dtype=int)]
        )
        scores = self._compute_attack_scores(X_test, synth, ref)
        return labels, scores

    # ------------------------------------------------------------------
    # Evaluation helper (used by MIAEvaluator)
    # ------------------------------------------------------------------

    def eval(
        self,
        true_labels: np.ndarray,
        predicted_scores: np.ndarray,
        use_decision_metrics: bool = False,
    ) -> Dict[str, float]:
        """Evaluate attack performance.

        Args:
            true_labels:       Ground-truth binary membership labels.
            predicted_scores:  Continuous membership scores from attack().
            use_decision_metrics: Whether to also compute accuracy/precision/recall.

        Returns:
            Dictionary of metric_name → float.
        """
        evaluator = AttackEvaluator(true_labels, predicted_scores)
        results: Dict[str, float] = evaluator.roc_metrics()
        if use_decision_metrics:
            results.update(evaluator.classification_metrics())
            results.update(evaluator.privacy_metrics())
        return results
