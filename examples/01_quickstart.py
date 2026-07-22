"""Minimal DOCAS: fix a confounded positive treatment effect."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from docas import Aligner, audit_curve


def train(X, y, sample_weight=None):
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=80, random_state=0)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def main():
    rng = np.random.default_rng(0)
    n = 1_500
    severity = rng.normal(size=n)
    treatment = 0.8 * severity + rng.normal(scale=0.4, size=n)
    y = 1.2 * severity - 0.6 * treatment + rng.normal(scale=0.25, size=n)
    X = np.column_stack([rng.normal(size=n), treatment, rng.normal(size=n)])

    def target(X_rows, u, baseline):
        return baseline.predict(X_rows) * (1.0 - 0.4 * u)

    baseline = train(X, y)
    result = Aligner(
        train_fn=train,
        intervention_idx=1,
        target=target,
        context=lambda _X, yhat: yhat > np.median(yhat),
        span=1.0,
        mode="channel",
        n_anchors=300,
        n_u=11,
        synth_weight=25.0,
    ).fit(X, y, baseline=baseline)

    u = np.linspace(0, 1, 6)
    hi = X[baseline.predict(X) > np.median(baseline.predict(X))]
    base_curve = audit_curve(baseline, hi, intervention_idx=1, u=u, span=1.0, mode="add")
    aligned = audit_curve(result.model_, hi, intervention_idx=1, u=u, span=1.0, mode="channel")

    print(f"synthetic rows : {result.n_synth}")
    print(f"align. error   : {result.alignment_error:.4f}")
    print(f"baseline curve : {np.round(base_curve, 3)}")
    print(f"DOCAS curve    : {np.round(aligned, 3)}")


if __name__ == "__main__":
    main()
