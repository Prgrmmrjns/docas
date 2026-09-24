"""Synthetic confounded glucose / insulin toy data (no real patient records).

Severity drives both high treatment and high outcome, so an observational
regressor learns the wrong treatment effect. Useful for README / unit demos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticGlucose:
    """Toy panel for dose–response demos.

    Columns of ``X`` are current glucose, the treatment dose
    (``treatment_idx``), and the meal load. ``y`` is the change in glucose
    over a short horizon (negative means lowering).
    """

    X: np.ndarray
    y: np.ndarray
    treatment_idx: int = 1
    feature_names: tuple[str, ...] = ("glucose", "insulin", "carbs")

    @property
    def n_samples(self) -> int:
        return int(len(self.y))


def make_synthetic_glucose(
    n: int = 2_000,
    *,
    seed: int = 0,
    noise: float = 8.0,
    true_treatment_effect: float = -25.0,
) -> SyntheticGlucose:
    """Draw a confounded synthetic glucose panel.

    Parameters
    ----------
    n :
        Number of rows.
    seed :
        RNG seed.
    noise :
        Gaussian noise std on ``y`` (outcome units).
    true_treatment_effect :
        True interventional slope of treatment → Δglucose at unit dose
        (negative = treatment lowers the outcome). Observational association
        is typically *positive* because of severity confounding.
    """
    rng = np.random.default_rng(int(seed))
    n = int(n)
    # Latent severity: drives both dosing and high glucose.
    severity = rng.normal(0.0, 1.0, n)
    glucose = 160.0 + 35.0 * severity + rng.normal(0.0, 12.0, n)
    carbs = np.clip(40.0 + 15.0 * severity + rng.normal(0.0, 10.0, n), 0.0, None)
    # Clinician gives more treatment when severity / glucose is high.
    insulin = np.clip(
        0.35 * severity + 0.015 * (glucose - 140.0) + rng.normal(0.0, 0.35, n),
        0.0,
        None,
    )
    # True physics: treatment lowers glucose; carbs raise it; severity raises it.
    y = (
        8.0 * severity
        + 0.15 * carbs
        + float(true_treatment_effect) * insulin
        + rng.normal(0.0, float(noise), n)
    )
    X = np.column_stack([glucose, insulin, carbs]).astype(float)
    return SyntheticGlucose(X=X, y=y.astype(float), treatment_idx=1)
