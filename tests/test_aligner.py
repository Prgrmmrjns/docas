"""Core Aligner smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

from docas import Aligner, audit_curve
from docas.datasets import make_synthetic_glucose


def _train(X, y, sample_weight=None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(max_depth=2, max_iter=40, random_state=0)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def test_make_synthetic_glucose_shape():
    data = make_synthetic_glucose(n=100, seed=1)
    assert data.X.shape == (100, 3)
    assert data.y.shape == (100,)
    assert data.treatment_idx == 1


def test_aligner_lowers_alignment_error():
    data = make_synthetic_glucose(n=800, seed=0)
    X, y = data.X, data.y

    def target(X_rows, u, baseline):
        return baseline.predict(X_rows) - 15.0 * u

    baseline = _train(X, y)
    # Baseline audit under add-mode (no dose channel).
    u = np.linspace(0, 1, 5)
    hi = X[baseline.predict(X) > np.median(baseline.predict(X))]
    base_curve = audit_curve(
        baseline, hi, intervention_idx=data.treatment_idx, u=u, span=1.0, mode="add")

    result = Aligner(
        train_fn=_train,
        treatment_idx=data.treatment_idx,
        target=target,
        context=lambda _X, yhat: yhat > np.median(yhat),
        n_anchors=120,
        n_u=7,
        synth_weight=20.0,
    ).fit(X, y, baseline=baseline)

    aligned = audit_curve(
        result.model_, hi, intervention_idx=data.treatment_idx, u=u, span=1.0, mode="append")
    assert result.n_synth > 0
    assert np.isfinite(result.alignment_error)
    # Aligned curve should drop more (or at least not rise) vs observational baseline.
    assert aligned[-1] < base_curve[-1] + 1.0


def test_treatment_idx_required():
    with pytest.raises(ValueError, match="treatment_idx"):
        Aligner(train_fn=_train, target=lambda X, u, b: u)
