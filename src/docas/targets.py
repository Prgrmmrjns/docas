"""Target DRC helpers (shape functions, not a fitted model)."""
from __future__ import annotations

import numpy as np


class IdentityScaler:
    """No-op scaler for unscaled features or labels."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, float)

    def inverse_transform(self, X):
        return np.asarray(X, float)


def interpolate(u_knots, drc_knots):
    """Piecewise-linear target DRC from knots. ``u`` is fractional in ``[0, 1]``."""
    uk = np.asarray(u_knots, float).ravel()
    dk = np.asarray(drc_knots, float).ravel()

    def drc(u):
        x = np.asarray(u, float)
        return np.interp(x.ravel(), uk, dk).reshape(x.shape)

    return drc


interp_target = interpolate


def linear_fraction(drop_at_full: float = 0.3):
    """``R(u) = drop_at_full · u``. Negative drop means the outcome falls."""
    drop = float(drop_at_full)

    def r(u):
        return drop * np.asarray(u, float)

    return r
