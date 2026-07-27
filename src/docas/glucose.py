"""Backward-compatible alias of :mod:`docas.t1d`."""
from docas.t1d import *  # noqa: F403
from docas.t1d import DOCAS, F0_REF, future, sample_synthetic, scale_delta_to_f0, target_delta_ref

__all__ = [
    "DOCAS",
    "F0_REF",
    "future",
    "sample_synthetic",
    "scale_delta_to_f0",
    "target_delta_ref",
]
