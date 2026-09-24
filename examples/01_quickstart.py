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


def target(u):
    return -20.0 * np.asarray(u, float)


def main():
    data = make_synthetic_glucose(n=2_000, seed=0)
    X, y = data.X, data.y
    ji = data.treatment_idx

    baseline = train(X, y)
    result = Aligner(
        train,
        treatment_idx=ji,
        target=target,
        span=1.0,
        synth_ratio=2.0,
        emp_frac=0.5,
    ).fit(X, y, baseline=baseline)

    u = np.linspace(0, 1, 6)
    print("synthetic rows :", result.n_synth)
    print("align. error   :", round(result.alignment_error, 4))
    print("baseline curve :", np.round(audit_curve(baseline, X, treatment_idx=ji, u=u), 2))
    print("aligned curve  :", np.round(audit_curve(result.model_, X, treatment_idx=ji, u=u), 2))


if __name__ == "__main__":
    main()
