"""
MuViS dataset loading + per-channel z-scoring + 90/10 stratified-random
train/val split.

Inputs come from the MuViS-DIR data bundle as ``train.ts`` / ``test.ts``
files in the .ts (sktime) format with multivariate float series and a
continuous regression target.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile
from torch.utils.data import Dataset


# ─── PyTorch dataset ──────────────────────────────────────────────────────────


class TimeSeriesDataset(Dataset):
    """Returns ``(x, y, weight)`` per sample for time-series regression.

    ``weights`` may be:
        None              → returns scalar weight 1.0
        np.ndarray (N,)   → returns scalar weight per sample
        list of arrays    → returns list of scalars (UVote multi-branch case)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, weights=None) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.feat_dim = X.shape[2]

        if weights is None:
            self.weights = None
            self._multi = False
        elif isinstance(weights, list):
            self.weights = [torch.tensor(w, dtype=torch.float32) for w in weights]
            self._multi = True
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self._multi = False

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.weights is None:
            w = torch.tensor(1.0)
        elif self._multi:
            w = [wb[idx] for wb in self.weights]
        else:
            w = self.weights[idx]
        return x, y, w


# ─── .ts loading ──────────────────────────────────────────────────────────────


def _load_ts(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a .ts file and return ``(X[N, T, C], y[N])`` as float32 arrays."""
    X, y = load_from_tsfile(
        path,
        return_data_type="numpy3d",
        return_y=True,
        y_dtype="float",
    )
    # sktime returns (N, C, T); transpose to MuViS convention (N, T, C).
    X = np.transpose(X, (0, 2, 1)).astype(np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y


# ─── End-to-end loader used by the trainer ────────────────────────────────────


def load_muvis_dataset(
    dataset_name: str,
    data_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
) -> dict:
    """Load and preprocess one MuViS dataset.

    Steps:
        1. Read ``<data_dir>/<dataset_name>/{train,test}.ts``.
        2. Stratified-random 90/10 split of train into train/val.
        3. Per-channel ``StandardScaler`` fit on training features only.

    Parameters
    ----------
    dataset_name : MuViS dataset id (e.g. ``"BeijingPM25Quality"`` or
        ``"REVS/2013_Monterey_Motorsports_Reunion"``).  Slashes are kept
        verbatim — they are part of the on-disk path.
    data_dir     : root containing per-dataset subdirectories.
    val_split    : fraction of training data used as validation.
    seed         : controls both the train/val split and (downstream) the
        Torch model init / data shuffling.

    Returns
    -------
    dict with keys ``X_train, X_val, X_test, y_train, y_val, y_test,
    feat_dim, scaler``.  Feature arrays are scaled (N, T, C) float32; target
    arrays are unscaled (N,) float32.
    """
    base = Path(data_dir) / dataset_name
    X_full, y_full = _load_ts(str(base / "train.ts"))
    X_test, y_test = _load_ts(str(base / "test.ts"))

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=val_split, random_state=seed, shuffle=True
    )

    feat_dim = int(X_train.shape[2])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, feat_dim)).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, feat_dim)).reshape(X_val.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, feat_dim)).reshape(X_test.shape)

    return {
        "X_train": X_train_s.astype(np.float32),
        "X_val": X_val_s.astype(np.float32),
        "X_test": X_test_s.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "feat_dim": feat_dim,
        "scaler": scaler,
    }
