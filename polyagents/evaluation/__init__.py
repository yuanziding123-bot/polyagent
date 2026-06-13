"""Evaluation subsystem — calibration / skill vs the market baseline (a first-class
peer to the L1-L4 pipeline: without it, the system can't be trusted live)."""
from __future__ import annotations

from .evaluate import categorize, evaluate, format_report
from .metrics import brier_score, calibration_curve, ece, log_loss

__all__ = [
    "evaluate",
    "format_report",
    "categorize",
    "brier_score",
    "log_loss",
    "ece",
    "calibration_curve",
]
