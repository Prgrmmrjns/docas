"""Any sklearn regressor factory."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from docas import Aligner, linear_fraction


def make_train(alpha=1.0):
    def train(X, y, sample_weight=None):
        m = Ridge(alpha=alpha)
        try:
            m.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            m.fit(X, y)
        return m

    return train


def main():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 3))
    y = 0.5 * X.sum(1) + rng.normal(scale=0.2, size=600)
    result = Aligner(
        make_train(alpha=0.5),
        treatment_idx=0,
        target=linear_fraction(-0.25),
        span=1.0,
        synth_ratio=1.0,
    ).fit(X, y)
    print(result.model_.predict(X[:3]))
    print("alignment_error", round(result.alignment_error, 4))


if __name__ == "__main__":
    main()
