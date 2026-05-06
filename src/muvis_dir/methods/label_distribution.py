"""
LDS, SQInv, and UVote per-sample weight computation.

Continuous targets are discretised into ``num_bins`` equal-width bins
spanning ``[min(targets), max(targets)]`` over the training set; the
DIR-review weight transformations are applied bin-wise.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang


def get_lds_kernel_window(kernel: str = "gaussian", ks: int = 5, sigma: float = 2.0) -> np.ndarray:
    """Build the 1-D LDS smoothing window of length ``ks``."""
    assert kernel in ("gaussian", "triang", "laplace"), kernel
    half_ks = (ks - 1) // 2

    if kernel == "gaussian":
        base = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        base = np.array(base, dtype=np.float64)
        window = gaussian_filter1d(base, sigma=sigma)
        window = window / window.max()
    elif kernel == "triang":
        window = np.array(triang(ks), dtype=np.float64)
    else:  # laplace
        xs = np.arange(-half_ks, half_ks + 1)
        laplace = np.exp(-np.abs(xs) / sigma) / (2.0 * sigma)
        window = laplace / laplace.max()

    return window.astype(np.float32)


def _assign_bins(targets: np.ndarray, num_bins: int) -> np.ndarray:
    """Map continuous targets to integer bin indices in [0, num_bins-1]."""
    t_min, t_max = float(targets.min()), float(targets.max())
    if t_min == t_max:
        return np.zeros(len(targets), dtype=np.int64)
    edges = np.linspace(t_min, t_max, num_bins + 1)
    bins = np.searchsorted(edges[1:], targets, side="left")
    return np.clip(bins, 0, num_bins - 1).astype(np.int64)


def compute_sqinv_weights(targets: np.ndarray, num_bins: int = 100) -> np.ndarray:
    """SQInv weights: ``weight[i] = 1 / sqrt(count[bin[i]])``, mean-normalised."""
    bins = _assign_bins(targets, num_bins)
    counts = {k: 0 for k in range(num_bins)}
    for b in bins:
        counts[int(b)] += 1
    counts = {k: float(np.sqrt(v)) for k, v in counts.items()}

    num_per_label = np.array([counts[int(b)] for b in bins], dtype=np.float32)
    num_per_label = np.maximum(num_per_label, 1e-8)

    weights = 1.0 / num_per_label
    scaling = float(len(weights)) / float(weights.sum())
    return (scaling * weights).astype(np.float32)


def compute_lds_weights(
    targets: np.ndarray,
    kernel: str = "gaussian",
    ks: int = 5,
    sigma: float = 2.0,
    reweight: str = "sqrt_inv",
    num_bins: int = 100,
) -> np.ndarray:
    """Label Distribution Smoothing weights.

    Steps (faithful to AgeDB ``_prepare_weights`` with ``lds=True``):
        1. histogram counts per bin
        2. transform counts (sqrt_inv or inverse-with-clip[5,1000])
        3. convolve transformed counts with the LDS kernel
        4. ``weight[i] = 1 / smoothed_count[bin[i]]``, mean-normalised
    """
    assert reweight in ("sqrt_inv", "inverse"), reweight

    bins = _assign_bins(targets, num_bins)
    counts = {k: 0 for k in range(num_bins)}
    for b in bins:
        counts[int(b)] += 1

    if reweight == "sqrt_inv":
        counts = {k: float(np.sqrt(v)) for k, v in counts.items()}
    else:
        counts = {k: float(np.clip(v, 5, 1000)) for k, v in counts.items()}

    kernel_window = get_lds_kernel_window(kernel, ks, sigma)
    values = np.array([counts[k] for k in range(num_bins)], dtype=np.float32)
    smoothed = convolve1d(values, weights=kernel_window, mode="constant")

    num_per_label = np.array([smoothed[int(b)] for b in bins], dtype=np.float32)
    num_per_label = np.maximum(num_per_label, 1e-8)

    weights = 1.0 / num_per_label
    scaling = float(len(weights)) / float(weights.sum())
    return (scaling * weights).astype(np.float32)


def compute_uvote_weights(
    targets: np.ndarray,
    num_branch: int = 2,
    num_bins: int = 100,
) -> list[np.ndarray]:
    """Per-expert per-sample weights for UVote.

    For each expert k with reweighting strength ``r_k`` linear-spaced in [0, 1],
    weight ∝ count[bin]^(-r_k); for ``r >= 1`` counts are clipped to [5, 1000].
    """
    bins = _assign_bins(targets, num_bins)
    raw_counts = {k: 0 for k in range(num_bins)}
    for b in bins:
        raw_counts[int(b)] += 1

    rs = np.linspace(0, 1, num=num_branch)
    all_weights = []

    for r in rs:
        vd: dict[int, float] = {}
        for k, v in raw_counts.items():
            vd[k] = 1.0 if v == 0 else float(np.power(float(v), float(r)))
        if r >= 1.0:
            vd = {k: float(np.clip(v, 5.0, 1000.0)) for k, v in vd.items()}

        num_per_label = np.array([vd[int(b)] for b in bins], dtype=np.float32)
        num_per_label = np.maximum(num_per_label, 1e-8)

        weights = 1.0 / num_per_label
        scaling = float(len(weights)) / float(weights.sum())
        all_weights.append((scaling * weights).astype(np.float32))

    return all_weights
