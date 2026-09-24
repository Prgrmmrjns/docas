"""Ceteris-paribus probes of a treatment column."""
from __future__ import annotations

import numpy as np


class AppendModel:
    """Wrap a model trained with an extra dose column (last feature)."""

    def __init__(self, model, n_features: int):
        self.model = model
        self.n_features = int(n_features)

    def predict(self, X):
        X = np.asarray(X, float)
        if X.shape[1] == self.n_features:
            X = np.column_stack([X, np.zeros(len(X))])
        return self.model.predict(X)

    def predict_intervention(self, X, u, span):
        X, u = np.asarray(X, float), np.asarray(u, float).ravel()
        rows = np.repeat(X, u.size, 0)
        rows = np.column_stack([rows, np.tile(u * float(span), len(X))])
        return self.model.predict(rows).reshape(len(X), u.size)


def apply_dose(X, *, treatment_idx: int, u, span: float = 1.0, mode: str = "replace"):
    """Rewrite treatment on copies of ``X``. ``u`` is fractional in ``[0, 1]``."""
    X = np.asarray(X, float)
    u = np.asarray(u, float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2-d")
    if mode not in {"replace", "add", "append"}:
        raise ValueError("mode must be 'replace', 'add', or 'append'")
    if len(X) == 0:
        extra = 1 if mode == "append" else 0
        return np.zeros((0, X.shape[1] + extra), float)
    if u.size not in (1, len(X)):
        raise ValueError("u must be scalar or one value per row")
    dose = np.broadcast_to(u * float(span), (len(X),)).astype(float)
    if mode == "append":
        return np.column_stack([X, dose])
    out = X.copy()
    if mode == "replace":
        out[:, treatment_idx] = dose
    else:
        out[:, treatment_idx] = out[:, treatment_idx] + dose
    return out


def probe(model, X, *, treatment_idx: int, u, span: float = 1.0, mode: str = "replace"):
    """Predictions under a dose grid. Shape ``(n, |u|)``."""
    X = np.asarray(X, float)
    u = np.asarray(u, float).ravel()
    if len(X) == 0:
        return np.zeros((0, u.size), float)
    if hasattr(model, "predict_intervention"):
        return np.asarray(model.predict_intervention(X, u, span), float)
    rows = np.repeat(X, u.size, 0)
    uu = np.tile(u, len(X))
    rows = apply_dose(rows, treatment_idx=treatment_idx, u=uu, span=span, mode=mode)
    return np.asarray(model.predict(rows), float).reshape(len(X), u.size)


def audit_curve(model, X, *, treatment_idx: int | None = None, intervention_idx: int | None = None, u=None, span: float = 1.0, mode: str = "replace"):
    """Partial dependence: mean ICE over rows of ``X`` (shape ``(|u|,)``)."""
    ji = int(treatment_idx if treatment_idx is not None else intervention_idx)
    u = np.linspace(0.0, 1.0, 11) if u is None else np.asarray(u, float).ravel()
    return probe(model, X, treatment_idx=ji, u=u, span=span, mode=mode).mean(0)


def centred_ice(model, X, *, treatment_idx: int, u=None, span: float = 1.0, mode: str = "replace"):
    """ICE minus the prediction at null dose. Shape ``(n, |u|)``."""
    u = np.linspace(0.0, 1.0, 11) if u is None else np.asarray(u, float).ravel()
    pred = probe(model, X, treatment_idx=treatment_idx, u=u, span=span, mode=mode)
    if u.size and abs(float(u[0])) < 1e-12:
        return pred - pred[:, :1]
    zero = probe(model, X, treatment_idx=treatment_idx, u=np.zeros(1), span=span, mode=mode)
    return pred - zero
