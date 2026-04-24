"""
Kernel Density Estimation (KDE) Membership Inference Attack.

Attack logic:
    Fit a KDE on the synthetic/CF proxy dataset, then score each test point
    by its density under that KDE.  Points with higher density are more likely
    to be members (they lie in regions densely covered by the proxy data).

    score(x) = KDE_synth(x)

This is a lightweight alternative to DOMIAS for cases where BNAF is unavailable.
"""

from typing import Optional

import numpy as np
from scipy import stats

from ..base import BaseAttacker


class density_estimate(BaseAttacker):
    """KDE-based density membership inference attack.

    Hyper-parameters
    ----------------
    bw_method : str or float
        Bandwidth selector for scipy.stats.gaussian_kde.
        Common values: ``'silverman'``, ``'scott'``, or a float scalar.
        Default: ``'silverman'``.
    """

    def __init__(self, hyper_parameters=None) -> None:
        defaults = {"bw_method": "silverman"}
        self.hyper_parameters = {**defaults, **(hyper_parameters or {})}
        super().__init__(self.hyper_parameters)
        self.name = "DensityEstimate"

    def _compute_attack_scores(
        self,
        X_test: np.ndarray,
        synth: np.ndarray,
        ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate KDE(synth) at every test point.

        Args:
            X_test: Test points, shape (N, d).
            synth:  Proxy/CF dataset used to fit the KDE, shape (M, d).
            ref:    Unused; kept for interface consistency.

        Returns:
            Density scores, shape (N,).
        """
        bw = self.hyper_parameters["bw_method"]
        # gaussian_kde requires data in (d, n) format
        kde = stats.gaussian_kde(synth.T, bw_method=bw)
        return kde.evaluate(X_test.T)
