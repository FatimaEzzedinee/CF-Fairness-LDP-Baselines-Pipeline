"""
Distance to Closest Record (DCR) Membership Inference Attack.

Reference:
    Stadler, T., Oprisanu, B., and Troncoso, C.
    Synthetic data – anonymisation groundhog day.
    USENIX Security Symposium, 2022.

Attack logic:
    A test point x is deemed a *member* when it is unusually close to the
    synthetic/CF proxy dataset.  The membership score is the *negative*
    minimum-distance from x to synth (higher score ↔ closer ↔ more likely member).
"""

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..base import BaseAttacker


class dcr(BaseAttacker):
    """Distance to Closest Record (DCR) attack.

    Hyper-parameters
    ----------------
    distance_type : int
        Minkowski p-norm used for distances.
        1 = Manhattan (L1), 2 = Euclidean (L2).  Default: 2.
    """

    def __init__(self, hyper_parameters=None) -> None:
        defaults = {"distance_type": 2}
        self.hyper_parameters = {**defaults, **(hyper_parameters or {})}
        super().__init__(self.hyper_parameters)
        self.name = "DCR"

    def _compute_attack_scores(
        self,
        X_test: np.ndarray,
        synth: np.ndarray,
        ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Score each test point by its negative distance to the closest synth record.

        Args:
            X_test: Test points (members + non-members), shape (N, d).
            synth:  Proxy/CF dataset, shape (M, d).
            ref:    Unused in DCR; kept for interface consistency.

        Returns:
            Scores array of shape (N,).  Higher score ↔ closer to synth ↔ likely member.
        """
        tree = cKDTree(synth)
        distances, _ = tree.query(
            X_test, k=1, p=self.hyper_parameters["distance_type"]
        )
        # Negate: small distance → high score (member signal)
        return -distances
