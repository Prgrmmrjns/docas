"""README quickstart: align on synthetic confounded glucose data."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from docas import Aligner, audit_curve
from docas.datasets import make_synthetic_glucose


def train(X, y, sample_weight=None):
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=80, random_state=0)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def target(X_rows, u, baseline):
    return baseline.predict(X_rows) - 20.0 * u


def main():
    data = make_synthetic_glucose(n=2_000, seed=0)
    X, y = data.X, data.y

    baseline = train(X, y)
    result = Aligner(
        train_fn=train,
        treatment_idx=data.treatment_idx,
        target=target,
        context=lambda _X, yhat: yhat > np.median(yhat),
        span=1.0,
        mode="append",
        n_anchors=300,
        n_u=11,
        synth_weight=25.0,
    ).fit(X, y, baseline=baseline)

    u = np.linspace(0, 1, 6)
    hi = X[baseline.predict(X) > np.median(baseline.predict(X))]
    base_curve = audit_curve(
        baseline, hi, intervention_idx=data.treatment_idx, u=u, span=1.0, mode="add")
    aligned = audit_curve(
        result.model_, hi, intervention_idx=data.treatment_idx, u=u, span=1.0, mode="append")

    print(f"synthetic rows : {result.n_synth}")
    print(f"align. error   : {result.alignment_error:.4f}")
    print(f"baseline curve : {np.round(base_curve, 2)}")
    print(f"aligned curve  : {np.round(aligned, 2)}")


if __name__ == "__main__":
    main()
