"""
BNAF density estimation stub.

The original BNAF (Block Neural Autoregressive Flow) implementation is an
optional dependency that is not included in this repository.  This stub
replaces it with a KDE-based fallback so that gen_lra and domias can still
run without error.

If you have the real BNAF package available, replace the functions below with
the actual implementations.
"""

import warnings
from typing import Any, Tuple

import numpy as np
from scipy import stats


def density_estimator_trainer(
    data: np.ndarray,
    epochs: int = 100,
    save: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """KDE stand-in for the BNAF density estimator trainer.

    Args:
        data:   Training data, shape (N, d).
        epochs: Ignored (BNAF parameter kept for interface compatibility).
        save:   Ignored.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        (trainer_placeholder, kde_model)
        The second element is the fitted scipy gaussian_kde object, which is
        what compute_log_p_x expects.
    """
    warnings.warn(
        "BNAF is not installed; falling back to KDE density estimation.",
        ImportWarning,
        stacklevel=2,
    )
    kde = stats.gaussian_kde(data.T, bw_method="silverman")
    return None, kde  # trainer placeholder is None


def compute_log_p_x(model: Any, X: Any) -> Any:
    """Evaluate log-density for each row of X under the fitted KDE model.

    Args:
        model: The object returned as the second element of
               density_estimator_trainer() — here a scipy gaussian_kde.
        X:     Data tensor/array of shape (N, d).
               Accepts both torch.Tensor and np.ndarray.

    Returns:
        Log-density values as a numpy array (or torch tensor if torch is used
        by the caller).  Shape: (N,).
    """
    # Accept torch tensors transparently
    try:
        import torch
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
            log_p = np.log(model.evaluate(X_np.T) + 1e-20)
            return torch.tensor(log_p, dtype=torch.float32)
    except ImportError:
        pass

    X_np = np.asarray(X)
    return np.log(model.evaluate(X_np.T) + 1e-20)
