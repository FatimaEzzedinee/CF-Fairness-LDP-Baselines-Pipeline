"""
Monte-Carlo (MC) Membership Inference Attack.

Reference:
    Inspired by:  van Breugel et al., "Membership inference attacks against
    synthetic data through overfitting detection", AISTATS 2023.

Attack logic:
    The MC attacker compares the distance of each test point to the synthetic
    (proxy/CF) dataset vs. to the reference (population) dataset.

    score(x) = dist(x, ref) − dist(x, synth)

    A positive score indicates x is closer to the synthetic data than to the
    reference population, which is a signal of membership.
"""

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..base import BaseAttacker


class mc(BaseAttacker):
    """Monte-Carlo differential distance MIA.

    Hyper-parameters
    ----------------
    distance_type : int
        Minkowski p-norm.  1 = L1, 2 = L2.  Default: 2.
    """

    def __init__(self, hyper_parameters=None) -> None:
        defaults = {"distance_type": 2}
        self.hyper_parameters = {**defaults, **(hyper_parameters or {})}
        super().__init__(self.hyper_parameters)
        self.name = "MC"

    def _compute_attack_scores(
        self,
        X_test: np.ndarray,
        synth: np.ndarray,
        ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute dist(x, ref) − dist(x, synth) for each test point.

        Args:
            X_test: Test points, shape (N, d).
            synth:  Proxy/CF dataset, shape (M, d).
            ref:    Reference/population dataset, shape (R, d).
                    If None, falls back to −dist(x, synth) (equivalent to DCR).

        Returns:
            Scores array of shape (N,).
        """
        p = self.hyper_parameters["distance_type"]
        synth_tree = cKDTree(synth)
        dist_synth, _ = synth_tree.query(X_test, k=1, p=p)

        if ref is not None and len(ref) > 0:
            ref_tree = cKDTree(ref)
            dist_ref, _ = ref_tree.query(X_test, k=1, p=p)
            # Points closer to synth than to ref → higher score → more likely member
            return dist_ref - dist_synth
        else:
            # Fallback: no reference dataset — behave like DCR
            return -dist_synth
