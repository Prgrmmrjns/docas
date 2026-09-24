"""DOCAS — dose–response curve alignment via synthetic augmentation."""

from __future__ import annotations

from docas.aligner import Aligner, AlignResult, eval_target, sample_synthetic, temporal_split
from docas.intervene import AppendModel as InterventionModel, apply_dose, audit_curve, centred_ice, probe
from docas.targets import IdentityScaler, interpolate as interp_target, linear_fraction

__version__ = "0.3.0"

__all__ = [
    "Aligner",
    "AlignResult",
    "IdentityScaler",
    "InterventionModel",
    "apply_dose",
    "audit_curve",
    "centred_ice",
    "eval_target",
    "interp_target",
    "linear_fraction",
    "probe",
    "sample_synthetic",
    "temporal_split",
    "__version__",
]
