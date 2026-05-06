"""
Balanced regression metrics for the per-(method, dataset) results table.

Definitions
-----------
Binning (Freedman--Diaconis rule on ``y_test``):
    w_FD  = 2 * IQR(y) * N^(-1/3)
    K     = ceil((max - min) / w_FD)              fallback: Sturges if IQR == 0
    edges = linspace(min, max, K+1)               equal-width bins

bMAE:
    (1/K_eff) * sum_{k: n_k>0} (1/n_k) * sum_{i in bin k} |y_i - yhat_i|

bMASE:
    bMAE(model) / bMAE(c_ref)   with c_ref = median(y_train)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.stats import iqr as scipy_iqr

logger = logging.getLogger(__name__)


def compute_bins(y: np.ndarray) -> np.ndarray:
    """Freedman--Diaconis bin edges for array ``y``."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    a, b = y.min(), y.max()

    if a == b:
        warnings.warn(f"All target values are identical ({a}). Using single bin.", UserWarning)
        return np.array([a - 0.5, b + 0.5])

    iqr_val = scipy_iqr(y)

    if iqr_val == 0:
        K = max(1, int(np.ceil(np.log2(n))) + 1)
        logger.warning("IQR=0; falling back to Sturges rule (K=%d).", K)
    else:
        w_fd = 2.0 * iqr_val * (n ** (-1.0 / 3.0))
        K = max(1, int(np.ceil((b - a) / w_fd)))

    return np.linspace(a, b, K + 1)


def assign_bins(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values to bin indices in ``[0, K-1]``; right-closed last bin."""
    return np.digitize(y, edges[1:-1])


def bMAE(y_true: np.ndarray, y_pred: np.ndarray, edges: np.ndarray) -> float:
    """Macro-average of per-bin MAE (equal weight per non-empty bin)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    bin_idx = assign_bins(y_true, edges)
    K = len(edges) - 1

    per_bin = []
    for k in range(K):
        mask = bin_idx == k
        if mask.sum() == 0:
            continue
        per_bin.append(np.mean(np.abs(y_true[mask] - y_pred[mask])))

    return float(np.mean(per_bin)) if per_bin else float("nan")


def bMASE(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    edges: np.ndarray,
) -> float:
    """``bMAE(model) / bMAE(c_ref)`` with ``c_ref = median(y_train))``."""
    c_ref = float(np.median(y_train))
    bmae_model = bMAE(y_true, y_pred, edges)
    bmae_ref = bMAE(y_true, np.full_like(y_true, c_ref, dtype=float), edges)

    if bmae_ref == 0 or np.isnan(bmae_ref):
        return float("nan")
    return float(bmae_model / bmae_ref)
