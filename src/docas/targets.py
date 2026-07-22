"""Target-function helpers."""
from __future__ import annotations

import numpy as np


class IdentityScaler:
    """No-op scaler for unscaled features / labels."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, float)

    def inverse_transform(self, X):
        return np.asarray(X, float)


def interpolate(u_knots, tau_knots):
    """Piecewise-linear target ``τ(u)`` from knots (u in [0, 1])."""
    uk = np.asarray(u_knots, float).ravel()
    tk = np.asarray(tau_knots, float).ravel()

    def tau(u):
        x = np.asarray(u, float)
        return np.interp(x.ravel(), uk, tk).reshape(x.shape)

    return tau


def linear_fraction(drop_at_full: float = 0.3):
    """Fractional lowering ``R(u) = drop_at_full · u`` (R(0)=0, R(1)=drop)."""
    drop = float(drop_at_full)

    def r(u):
        return drop * np.asarray(u, float)

    return r
