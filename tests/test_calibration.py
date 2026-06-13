"""Tests for the calibration layer and the time-annualized edge gate."""
from __future__ import annotations

from pytest import approx

from polyagents.agents.calibration import (
    IdentityCalibrator,
    ShrinkageCalibrator,
    calibrator_from_config,
)
from polyagents.agents.decision_agent import decide
from polyagents.agents.risk import annualized_edge
from polyagents.agents.schemas import Signal
from polyagents.default_config import DEFAULT_CONFIG


def test_shrinkage_blends_toward_market():
    c = ShrinkageCalibrator(0.3)
    assert c.calibrate(0.70, 0.50) == approx(0.64)   # 0.7*0.7 + 0.3*0.5
    assert c.calibrate(0.30, 0.50) == approx(0.36)
    assert ShrinkageCalibrator(0.0).calibrate(0.7, 0.5) == 0.7   # trust model
    assert ShrinkageCalibrator(1.0).calibrate(0.7, 0.5) == 0.5   # trust market fully


def test_calibrator_from_config():
    assert isinstance(calibrator_from_config({"calibration_market_weight": 0.0}), IdentityCalibrator)
    assert isinstance(calibrator_from_config({"calibration_market_weight": 0.3}), ShrinkageCalibrator)


def test_annualized_edge_math():
    assert annualized_edge(0.10, 0.50, 36.5) == approx(2.0)     # (0.1/0.5)*(365/36.5)
    assert annualized_edge(0.06, 0.50, 365) == approx(0.12)


def test_apy_gate_blocks_long_dated_then_allows_short():
    cfg = DEFAULT_CONFIG.copy()
    cfg["calibration_market_weight"] = 0.0     # isolate the APY gate
    cfg["min_annualized_edge"] = 0.50
    cfg["max_spread_bps"] = 1000.0
    sig = Signal(direction="yes", p_true=0.58, conviction="high", rationale="r")

    long_dated = decide(sig, 0.50, 20000, 100, cfg, days_to_expiry=300)  # edge 8% but ~20% APY
    assert long_dated.action == "hold"
    assert any("APY" in r for r in long_dated.reasons)

    short_dated = decide(sig, 0.50, 20000, 100, cfg, days_to_expiry=10)  # same edge, big APY
    assert short_dated.action == "buy"


def test_decision_records_raw_and_calibrated():
    cfg = DEFAULT_CONFIG.copy()
    cfg["max_spread_bps"] = 1000.0
    sig = Signal(direction="yes", p_true=0.80, conviction="high", rationale="r")
    d = decide(sig, 0.50, 20000, 100, cfg, days_to_expiry=15)
    assert d.raw_p_true == 0.80
    assert d.p_true == approx(0.71)            # 0.8*0.7 + 0.5*0.3
    assert d.days_to_expiry == 15
