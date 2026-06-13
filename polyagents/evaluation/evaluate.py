"""Evaluation subsystem — does our p_true actually have edge over the market?

The headline question (per the feedback): a prediction market is, in many
domains, a well-calibrated baseline. If our model's probabilities don't beat
"just trust the market price", the apparent edge is noise. So we score the
model's predictions AND the market's, on the same resolved trades, and compare.

Operates over the decision log (memory records), including HOLDs — every analysed
market with a known outcome is a data point (counterfactual logging), not just
the ones we traded. Stratified by a coarse keyword category.
"""
from __future__ import annotations

from .metrics import brier_score, calibration_curve, ece, log_loss

_CATEGORIES = {
    "politics": ["election", "president", "senate", "vote", "minister", "parliament", "govern", "poll"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "token", "coin", "solana"],
    "sports": ["win the", "fifa", "world cup", "nba", "match", "vs.", " vs ", "league", "cup", "open"],
    "economy": ["fed", "rate", "inflation", "gdp", "cpi", "recession", "jobs", "tariff"],
    "geopolitics": ["ceasefire", "war", "sanction", "airspace", "peace", "nuclear", "border"],
}


def categorize(question: str) -> str:
    q = (question or "").lower()
    for cat, kws in _CATEGORIES.items():
        if any(k in q for k in kws):
            return cat
    return "other"


def _score_group(records: list[dict]) -> dict:
    """Score one group of resolved records against the market baseline."""
    y = [1.0 if r.get("won") else 0.0 for r in records]
    model = [float(r.get("p_true")) for r in records]                 # calibrated p used to size
    raw = [float(r.get("raw_p_true") if r.get("raw_p_true") is not None else r.get("p_true")) for r in records]
    market = [float(r.get("market_price")) for r in records]
    model_brier, market_brier = brier_score(model, y), brier_score(market, y)
    return {
        "n": len(records),
        "hit_rate": sum(y) / len(y),
        "model_brier": model_brier,
        "raw_brier": brier_score(raw, y),
        "market_brier": market_brier,
        "brier_skill_vs_market": (1 - model_brier / market_brier) if market_brier else 0.0,
        "beats_market": model_brier < market_brier,
        "model_log_loss": log_loss(model, y),
        "market_log_loss": log_loss(market, y),
        "model_ece": ece(model, y),
        "calibration_curve": calibration_curve(model, y),
    }


def evaluate(records: list[dict]) -> dict:
    """Overall + per-category scores over resolved records."""
    resolved = [
        r for r in records
        if r.get("status") == "resolved" and r.get("won") is not None
        and r.get("p_true") is not None and r.get("market_price") is not None
    ]
    if not resolved:
        return {"n": 0, "pending": sum(1 for r in records if r.get("status") == "pending")}
    by_cat: dict[str, list[dict]] = {}
    for r in resolved:
        by_cat.setdefault(categorize(r.get("question", "")), []).append(r)
    return {
        "overall": _score_group(resolved),
        "by_category": {cat: _score_group(rs) for cat, rs in sorted(by_cat.items())},
    }


def format_report(result: dict) -> str:
    if "overall" not in result:
        return f"No resolved trades to evaluate ({result.get('pending', 0)} pending)."
    o = result["overall"]
    verdict = "BEATS market ✅" if o["beats_market"] else "does NOT beat market ❌ (edge is likely noise)"
    lines = [
        f"Evaluation — {o['n']} resolved predictions",
        f"  model {verdict}",
        f"  Brier: model {o['model_brier']:.3f}  vs  market {o['market_brier']:.3f}  "
        f"(skill {o['brier_skill_vs_market']:+.1%})",
        f"  raw-model Brier {o['raw_brier']:.3f}  (calibration {'helped' if o['model_brier'] <= o['raw_brier'] else 'hurt'})",
        f"  log-loss: model {o['model_log_loss']:.3f} vs market {o['market_log_loss']:.3f}",
        f"  calibration error (ECE): {o['model_ece']:.3f}  |  hit rate {o['hit_rate']:.0%}",
        "  by category:",
    ]
    for cat, s in result["by_category"].items():
        flag = "✅" if s["beats_market"] else "❌"
        lines.append(f"    {cat:12} n={s['n']:<3} Brier {s['model_brier']:.3f} vs mkt {s['market_brier']:.3f} {flag}")
    return "\n".join(lines)
