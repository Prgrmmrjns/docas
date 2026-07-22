"""Custom target from interpolation knots."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from docas import Aligner, interp_target


def train(X, y, sample_weight=None):
    m = HistGradientBoostingRegressor(max_depth=3, random_state=1)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def main():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1_000, 4))
    y = X[:, 0] - 0.2 * X[:, 2] + rng.normal(scale=0.3, size=len(X))
    shape = interp_target([0.0, 0.4, 1.0], [0.0, 0.05, 0.35])

    def target(X_rows, u, baseline):
        return baseline.predict(X_rows) * (1.0 - shape(u))

    out = Aligner(
        train_fn=train,
        intervention_idx=2,
        target=target,
        context=lambda _X, yhat: yhat > 0.0,
        mode="add",
        span=2.0,
    ).fit(X, y)
    print("mode=add  n_synth=", out.n_synth, " alignment=", round(out.alignment_error, 4))


if __name__ == "__main__":
    main()
