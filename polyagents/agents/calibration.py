"""Calibration layer — adjust the raw LLM ``p_true`` before it drives Kelly.

Kelly sizing assumes you KNOW the true probability. An LLM's ``p_true`` is poorly
calibrated, and quarter-Kelly only defends against variance *under a known
probability* — not against ``p_true`` itself being biased. The prediction-market
price is, in many domains, a well-calibrated baseline, so the safest calibration
is to **shrink the model estimate toward the market price**: act only on the part
of your view the market doesn't already reflect, scaled by how much you trust the
model.

A data-driven calibrator (isotonic / Platt fit on realised outcomes) can replace
``ShrinkageCalibrator`` once enough resolved history exists; the evaluation
subsystem (Brier / ECE vs the market baseline) measures whether it actually helps.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Calibrator(Protocol):
    def calibrate(self, p_true: float, market_price: float) -> float:
        ...


class IdentityCalibrator:
    """No-op — trust the raw estimate (use only when you know it's calibrated)."""

    def calibrate(self, p_true: float, market_price: float) -> float:
        return p_true


class ShrinkageCalibrator:
    """``p_cal = (1-w)*p_true + w*market_price``; ``w`` = trust in the market baseline."""

    def __init__(self, market_weight: float = 0.3) -> None:
        self.w = max(0.0, min(1.0, market_weight))

    def calibrate(self, p_true: float, market_price: float) -> float:
        p = (1.0 - self.w) * p_true + self.w * market_price
        return max(0.0, min(1.0, p))


def calibrator_from_config(config: dict) -> Calibrator:
    w = float(config.get("calibration_market_weight", 0.0) or 0.0)
    return ShrinkageCalibrator(w) if w > 0 else IdentityCalibrator()
