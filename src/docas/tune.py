"""Validation helpers for tuning targets and model hyperparameters.

Use an inner split of the training partition. Never select on held-out test data.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def temporal_split(X, y, *, val_frac: float = 0.5):
    """Chronological train/validation cut (first 1−val_frac for fitting)."""
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    n = max(1, len(X) - max(1, int(round(len(X) * float(val_frac)))))
    return X[:n], y[:n], X[n:], y[n:]


def scale_target(shape_fn: Callable, amp: float = 1.0):
    """Return ``u ↦ amp · shape_fn(u)`` for amplitude calibration of τ."""
    a = float(amp)

    def scaled(u, *, _s=shape_fn, _a=a):
        return _a * np.asarray(_s(u), float)

    return scaled


def rmse(y_true, y_pred) -> float:
    err = np.asarray(y_true, float).ravel() - np.asarray(y_pred, float).ravel()
    return float(np.sqrt(np.mean(err ** 2)))


def objective_sum(rmse_val: float, align_val: float) -> float:
    """Default selection objective: validation RMSE + alignment error."""
    return float(rmse_val) + float(align_val)
