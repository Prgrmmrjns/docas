"""DOCAS — Dose-response Curve Alignment via Synthetic augmentation.

Append synthetic (context, dose, label) rows and retrain so the model's
interventional audit curve matches a user target τ(u).

    from docas import Aligner
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = Aligner(train_fn=..., intervention_idx=1, target=...).fit(X, y).model_
"""

from __future__ import annotations

from docas.aligner import Aligner, AlignResult
from docas.glucose import DOCAS, F0_REF, future, sample_synthetic, scale_delta_to_f0, target_delta_ref
from docas.intervene import InterventionModel, audit_curve, probe
from docas.targets import IdentityScaler, interpolate as interp_target, linear_fraction

__version__ = "0.1.0"

__all__ = [
    "Aligner",
    "AlignResult",
    "DOCAS",
    "F0_REF",
    "IdentityScaler",
    "InterventionModel",
    "audit_curve",
    "future",
    "interp_target",
    "linear_fraction",
    "probe",
    "sample_synthetic",
    "scale_delta_to_f0",
    "target_delta_ref",
    "__version__",
]
