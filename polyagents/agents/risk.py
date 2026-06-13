"""Deterministic risk + sizing for the decision agent.

The Merakku v3.0 plan folds risk into the decision agent ("风控嵌入决策"). We
keep it as pure, auditable math — fractional Kelly sizing plus hard gates —
rather than letting an LLM size positions. Constants mirror the polymarket
reference repo (6% edge floor, quarter Kelly, 5% position cap).
"""
from __future__ import annotations


def effective_market_price(state: dict) -> tuple[float, str]:
    """The price to evaluate against: the live order-book mid, else the snapshot.

    The Gamma ``market_price`` seeded at run start can lag the live CLOB book
    (we've seen 0.69 vs a 0.735 book mid), which inflates the apparent edge. The
    book mid is the real, tradeable reference, so prefer it when available.
    """
    ob = (state.get("raw", {}) or {}).get("orderbook", {}) or {}
    mid = ob.get("mid")
    if mid is not None:
        return float(mid), "live book mid"
    return float(state.get("market_price", 0.0) or 0.0), "market snapshot"


def annualized_edge(edge: float, market_price: float, days_to_expiry: float) -> float:
    """Annualised expected return on capital for holding to resolution.

    A binary share costs ``market_price`` and the trade's expected profit is
    ``edge`` per $1 of share, realised only at resolution. Return on capital ≈
    ``edge / market_price`` over ``days_to_expiry`` days, annualised (simple).
    A 6% edge resolving in 9 days is a very different trade from 9 months — this
    is what the time-aware gate and APY reporting use.
    """
    if market_price <= 1e-9:
        return 0.0
    days = max(float(days_to_expiry), 0.5)
    return (edge / market_price) * (365.0 / days)


def edge_for_side(p_true: float, market_price: float) -> float:
    """Edge from buying the analysed side: estimated prob minus its price.

    A YES/NO share costs ``market_price`` and pays $1 if the side resolves.
    Positive edge ⇒ underpriced ⇒ buy candidate; negative ⇒ overpriced.
    """
    return p_true - market_price


def kelly_fraction(p_true: float, market_price: float) -> float:
    """Full-Kelly stake fraction for a binary contract at ``market_price``.

    f* = (q - p) / (1 - p), clamped to [0, 1]. Zero when there's no positive
    edge or the price leaves no room (≈ 1.0).
    """
    denom = 1.0 - market_price
    if denom <= 1e-9:
        return 0.0
    f = (p_true - market_price) / denom
    return max(0.0, min(1.0, f))
