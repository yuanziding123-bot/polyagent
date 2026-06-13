"""Probabilistic-forecast metrics — Brier, log-loss, and calibration (ECE).

These score *probability quality*, not win/loss. A trader can be profitable and
badly calibrated (or vice versa); for a probability-driven system the calibration
metrics are what tell you whether ``p_true`` is trustworthy enough to size on.
Pure functions over (predictions, binary outcomes) — no deps.
"""
from __future__ import annotations

import math

EPS = 1e-6


def brier_score(preds: list[float], outcomes: list[float]) -> float:
    """Mean squared error of probabilities. Lower is better; 0.25 = always 0.5."""
    if not preds:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(preds, outcomes)) / len(preds)


def log_loss(preds: list[float], outcomes: list[float]) -> float:
    """Mean negative log-likelihood (clipped). Punishes confident wrong calls."""
    if not preds:
        return float("nan")
    total = 0.0
    for p, y in zip(preds, outcomes):
        p = min(1 - EPS, max(EPS, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(preds)


def calibration_curve(preds: list[float], outcomes: list[float], bins: int = 10) -> list[dict]:
    """Reliability bins: for each probability bucket, mean predicted vs realised."""
    buckets: list[dict] = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(preds) if (lo <= p < hi or (b == bins - 1 and p == 1.0))]
        if not idx:
            continue
        buckets.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": len(idx),
            "mean_pred": sum(preds[i] for i in idx) / len(idx),
            "mean_outcome": sum(outcomes[i] for i in idx) / len(idx),
        })
    return buckets


def ece(preds: list[float], outcomes: list[float], bins: int = 10) -> float:
    """Expected Calibration Error — count-weighted |mean_pred − mean_outcome|."""
    n = len(preds)
    if n == 0:
        return float("nan")
    curve = calibration_curve(preds, outcomes, bins)
    return sum(b["n"] * abs(b["mean_pred"] - b["mean_outcome"]) for b in curve) / n
