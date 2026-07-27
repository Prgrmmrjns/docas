"""Interventional probes: edit a treatment column or append a dose feature."""
from __future__ import annotations

import numpy as np


class InterventionModel:
    """Wrap a model trained with an extra dose column (last feature).

    Observational rows use dose ``0``; probes set dose to ``u * span``.
    """

    def __init__(self, model, n_features: int):
        self.model = model
        self.n_features = int(n_features)

    def predict(self, X):
        X = np.asarray(X, float)
        if X.shape[1] == self.n_features:
            X = np.column_stack([X, np.zeros(len(X))])
        return self.model.predict(X)

    def predict_intervention(self, X, u, span):
        X, u = np.asarray(X, float), np.asarray(u, float)
        rows = np.repeat(X, u.size, 0)
        rows = np.column_stack([rows, np.tile(u * float(span), len(X))])
        return self.model.predict(rows).reshape(len(X), u.size)


def probe(model, X, *, intervention_idx: int, u, span: float = 1.0, mode: str = "add"):
    """Counterfactual predictions under dose grid ``u`` ∈ [0, 1].

    Parameters
    ----------
    mode :
        ``"add"`` — add ``u * span`` to column ``intervention_idx``.
        ``"replace"`` — set that column to ``u * span``.
        ``"append"`` — use :meth:`InterventionModel.predict_intervention`
        (model must support it). ``"channel"`` is accepted as an alias.
    """
    X, u = np.asarray(X, float), np.asarray(u, float).ravel()
    if len(X) == 0:
        return np.zeros((0, u.size), float)
    if mode in ("append", "channel") or hasattr(model, "predict_intervention"):
        return np.asarray(model.predict_intervention(X, u, span), float)
    rows = np.repeat(X, u.size, 0)
    dose = np.tile(u * float(span), len(X))
    if mode == "replace":
        rows[:, intervention_idx] = dose
    elif mode == "add":
        rows[:, intervention_idx] = rows[:, intervention_idx] + dose
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'add', 'replace', or 'append'")
    return np.asarray(model.predict(rows), float).reshape(len(X), u.size)


def audit_curve(model, X, *, intervention_idx: int, u=None, span: float = 1.0, mode: str = "add"):
    """Mean interventional response over rows of ``X`` (shape ``(|u|,)``)."""
    u = np.linspace(0.0, 1.0, 11) if u is None else np.asarray(u, float).ravel()
    return probe(model, X, intervention_idx=intervention_idx, u=u, span=span, mode=mode).mean(0)
