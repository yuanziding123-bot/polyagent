"""Decision agent — deterministic edge / Kelly / risk gates.

Reads the Signal (LLM probability read) plus market microstructure from the
state and produces a :class:`TradeDecision`. No LLM here: sizing and risk are
math so they're auditable and reproducible (the v3.0 plan embeds risk in the
decision agent rather than spinning up a separate risk Agent).
"""
from __future__ import annotations

from typing import Any, Callable

from .calibration import calibrator_from_config
from .risk import annualized_edge, edge_for_side, effective_market_price, kelly_fraction
from .schemas import Signal, TradeDecision

Node = Callable[[dict], dict]


def decide(
    signal: Signal,
    market_price: float,
    liquidity: float,
    spread_bps: float | None,
    config: dict,
    days_to_expiry: float = 30.0,
) -> TradeDecision:
    """Turn a signal into a sized, risk-gated decision.

    Two safeguards beyond raw edge: the model ``p_true`` is **calibrated** toward
    the market price before sizing (the market is a calibrated baseline; the LLM
    is noisy), and entries are gated on the **time-annualised** return — a 6%
    edge means nothing without knowing whether it resolves in days or months.
    """
    edge_floor = config["edge_floor"]
    bankroll = config["bankroll_usdc"]
    kelly_mult = config["kelly_multiplier"]
    max_frac = config["max_position_fraction"]
    min_liq = config["min_liquidity_usdc"]
    max_spread = config["max_spread_bps"]
    min_apy = config.get("min_annualized_edge", 0.0)

    raw_p = signal.p_true
    p_cal = calibrator_from_config(config).calibrate(raw_p, market_price)
    edge = edge_for_side(p_cal, market_price)
    apy = annualized_edge(edge, market_price, days_to_expiry)

    reasons: list[str] = []
    if abs(p_cal - raw_p) > 1e-4:
        reasons.append(f"p_true {raw_p:.2f} → calibrated {p_cal:.2f} (shrink to market)")

    gates: list[str] = []
    if liquidity < min_liq:
        gates.append(f"liquidity ${liquidity:,.0f} < ${min_liq:,.0f}")
    if spread_bps is not None and spread_bps > max_spread:
        gates.append(f"spread {spread_bps:.0f}bps > {max_spread:.0f}bps")

    def out(action: str, kelly: float, size: float) -> TradeDecision:
        return TradeDecision(action, p_cal, market_price, edge, kelly, size, reasons,
                             annualized_edge=apy, raw_p_true=raw_p, days_to_expiry=days_to_expiry)

    def hold(reason: str) -> TradeDecision:
        reasons.append(reason)
        return out("hold", 0.0, 0.0)

    if signal.direction == "none":
        return hold("signal: no directional lean")
    if abs(edge) < edge_floor:
        return hold(f"|edge| {abs(edge):.1%} < floor {edge_floor:.0%}")

    if edge >= edge_floor:
        if gates:
            return hold("edge present but risk gate(s): " + "; ".join(gates))
        if apy < min_apy:
            return hold(f"edge +{edge:.1%} but APY {apy:.0%} < {min_apy:.0%} "
                        f"({days_to_expiry:.0f}d to expiry — capital locked too long)")
        f = min(kelly_fraction(p_cal, market_price) * kelly_mult, max_frac)
        size = round(f * bankroll, 2)
        reasons.append(f"edge +{edge:.1%} (APY {apy:.0%}); {kelly_mult:g}x Kelly → {f:.2%} bankroll")
        if size <= 0:
            return hold("computed size rounds to $0")
        return out("buy", round(f, 4), size)

    # edge <= -edge_floor: overpriced — avoid / exit (no shorting on Polymarket).
    reasons.append(f"edge {edge:.1%} ≤ -{edge_floor:.0%}: overpriced, avoid/exit")
    return out("sell", 0.0, 0.0)


def _format_decision_report(d: TradeDecision) -> str:
    head = (f"DECISION: {d.action.upper()}  (p_cal {d.p_true:.2f} vs price {d.market_price:.2f}, "
            f"edge {d.edge:+.1%}, APY {d.annualized_edge:+.0%})")
    if d.action == "buy":
        head += f"  size ${d.size_usdc:,.2f} ({d.kelly_fraction:.2%} bankroll)"
    return head + "\n- " + "\n- ".join(d.reasons)


def create_decision_agent(config: dict) -> Node:
    def node(state: dict) -> dict[str, Any]:
        signal: Signal = state["signal"]
        raw = state.get("raw", {})
        ob = raw.get("orderbook", {}) or {}
        spread_bps = ob.get("spread_bps")
        liquidity = float(state.get("liquidity", 0.0) or 0.0)
        days = float(state.get("days_to_expiry", 30.0) or 30.0)
        price, price_source = effective_market_price(state)
        decision = decide(signal, price, liquidity, spread_bps, config, days_to_expiry=days)
        decision.reasons.insert(0, f"price {price:.3f} ({price_source})")
        return {"trade_decision": decision, "decision_report": _format_decision_report(decision)}

    return node
