"""Tests for the evaluation subsystem (calibration / skill vs market baseline)."""
from __future__ import annotations

from pytest import approx

from polyagents.evaluation.evaluate import categorize, evaluate, format_report
from polyagents.evaluation.metrics import brier_score, ece, log_loss


def test_brier_and_log_loss():
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == approx(0.25)
    assert log_loss([0.9], [1.0]) < log_loss([0.1], [1.0])    # confident-right < confident-wrong


def test_ece_perfect_and_bad():
    assert ece([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]) == approx(0.0)
    assert ece([0.9, 0.9], [0.0, 0.0]) == approx(0.9)         # always says 0.9, always 0


def test_categorize():
    assert categorize("Will the President win the election?") == "politics"
    assert categorize("Will Bitcoin hit 100k by 2026?") == "crypto"
    assert categorize("Spurs vs. Knicks") == "sports"
    assert categorize("Will the Fed cut rates?") == "economy"
    assert categorize("random question") == "other"


def _rec(won, p, mkt, q):
    return {"status": "resolved", "won": won, "p_true": p, "raw_p_true": p,
            "market_price": mkt, "question": q}


def test_model_beats_market():
    recs = [_rec(True, 0.9, 0.5, "election A"), _rec(False, 0.1, 0.5, "crypto B")]
    res = evaluate(recs)
    o = res["overall"]
    assert o["beats_market"] is True
    assert o["model_brier"] < o["market_brier"]
    assert o["brier_skill_vs_market"] > 0
    assert "BEATS market" in format_report(res)


def test_model_loses_to_market():
    # model is anti-correlated -> worse than the 0.5 market baseline
    recs = [_rec(True, 0.1, 0.5, "q1"), _rec(False, 0.9, 0.5, "q2")]
    res = evaluate(recs)
    assert res["overall"]["beats_market"] is False
    assert "does NOT beat market" in format_report(res)


def test_evaluate_empty():
    assert evaluate([])["n"] == 0
    assert "No resolved trades" in format_report(evaluate([]))
