"""Core Aligner smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

from docas import Aligner, audit_curve, sample_synthetic
from docas.datasets import make_synthetic_glucose


def _train(X, y, sample_weight=None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(max_depth=2, max_iter=40, random_state=0)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _target(u):
    return -15.0 * np.asarray(u, float)


def test_make_synthetic_glucose_shape():
    data = make_synthetic_glucose(n=100, seed=1)
    assert data.X.shape == (100, 3)
    assert data.y.shape == (100,)
    assert data.treatment_idx == 1


def test_sample_synthetic_incremental():
    data = make_synthetic_glucose(n=200, seed=0)
    Xs, ys, meta = sample_synthetic(
        data.X, data.y, treatment_idx=1, target=_target, span=1.0, synth_ratio=0.5, rng=np.random.default_rng(0),
    )
    assert len(ys) == 100
    assert meta["n_synth"] == 100
    assert Xs.shape[1] == data.X.shape[1]


def test_aligner_lowers_alignment_error():
    data = make_synthetic_glucose(n=800, seed=0)
    X, y = data.X, data.y
    ji = data.treatment_idx
    baseline = _train(X, y)
    u = np.linspace(0, 1, 5)
    base_curve = audit_curve(baseline, X, treatment_idx=ji, u=u)
    result = Aligner(
        _train, treatment_idx=ji, target=_target, synth_ratio=2.0, emp_frac=0.5,
    ).fit(X, y, baseline=baseline)
    aligned = audit_curve(result.model_, X, treatment_idx=ji, u=u)
    assert result.n_synth > 0
    assert np.isfinite(result.alignment_error)
    assert aligned[-1] < base_curve[-1] + 1.0
    assert result.alignment_error < 20.0


def test_treatment_idx_required():
    with pytest.raises(ValueError, match="treatment_idx"):
        Aligner(_train, target=_target)


def test_ambient_target():
    data = make_synthetic_glucose(n=400, seed=2)
    X, y = data.X, data.y

    def target(X_rows, u):
        return -0.1 * X_rows[:, 0] * np.asarray(u, float)

    out = Aligner(_train, treatment_idx=1, target=target, synth_ratio=1.0).fit(X, y)
    assert np.isfinite(out.alignment_error)
