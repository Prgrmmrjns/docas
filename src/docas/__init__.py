"""DOCAS — Dose-response Curve Alignment via Synthetic augmentation.

Core API (domain-agnostic)::

    from docas import Aligner, audit_curve
    from docas.datasets import make_synthetic_glucose

Optional T1D research helpers live in :mod:`docas.t1d` and are re-exported
here for backward compatibility.
"""

from __future__ import annotations

from docas.aligner import Aligner, AlignResult
from docas.intervene import InterventionModel, audit_curve, probe
from docas.targets import IdentityScaler, interpolate as interp_target, linear_fraction
from docas.tune import objective_sum, scale_target, temporal_split

# Optional T1D / manuscript helpers (prefer Aligner for new projects).
from docas.t1d import DOCAS, F0_REF, future, sample_synthetic, scale_delta_to_f0, target_delta_ref

__version__ = "0.2.0"

__all__ = [
    "Aligner",
    "AlignResult",
    "IdentityScaler",
    "InterventionModel",
    "audit_curve",
    "interp_target",
    "linear_fraction",
    "objective_sum",
    "probe",
    "scale_target",
    "temporal_split",
    # optional T1D
    "DOCAS",
    "F0_REF",
    "future",
    "sample_synthetic",
    "scale_delta_to_f0",
    "target_delta_ref",
    "__version__",
]
