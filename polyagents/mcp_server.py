"""polyagents MCP server — the trading engine exposed as MCP tools.

This is how polyagents plugs into an Alpha DevBox-style host: the platform's
chat agent (Claude) connects to this MCP server and calls these tools to scan
markets, pull data, size positions, paper-trade, and review P&L. The **agent**
does the reasoning ("is this side mispriced?"); polyagents provides the
**deterministic** capabilities (data, microstructure, factors, Kelly sizing,
paper execution, portfolio, settlement, RAG). No internal LLM / key is needed by
these tools.

Run it:
    python -m polyagents.mcp_server            # stdio (for Claude / Alpha DevBox)
    python -m polyagents.mcp_server --http     # streamable-http on :8000

Register with a host that reads `.mcp.json` (e.g. Claude Code / the Polymarket
docs MCP pattern):
    { "mcpServers": { "polyagents": { "command": "python",
        "args": ["-m", "polyagents.mcp_server"] } } }

Add new skills by exposing more @mcp.tool() functions here and documenting the
workflow in a SKILL.md under ./skills/ — see skills/README.md.
"""
from __future__ import annotations

import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from polyagents.agents.decision_agent import decide
from polyagents.agents.schemas import Signal, TradeDecision
from polyagents.dataflows.microstructure import compute_microstructure
from polyagents.dataflows.types import Market
from polyagents.execution.types import ExecutionResult, Order
from polyagents.graph.orchestrator import PolyAgentsGraph

mcp = FastMCP("polyagents")

# --- engine (one per server process; portfolio/RAG persist across tool calls) ---

_ENGINE: PolyAgentsGraph | None = None
_MARKETS: dict[str, Market] = {}   # token_id -> Market (populated by scans)


def engine() -> PolyAgentsGraph:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PolyAgentsGraph()
    return _ENGINE


def set_engine(e: PolyAgentsGraph | None) -> None:
    """Inject an engine (tests) and reset the market cache."""
    global _ENGINE, _MARKETS
    _ENGINE = e
    _MARKETS = {}


def _refresh_markets() -> None:
    eng = engine()
    raw = eng.client.list_active_markets(limit=eng.config["markets_limit"])
    for m in eng.client.to_markets(raw):
        _MARKETS[m.token_id] = m


def _get_market(token_id: str) -> Market | None:
    if token_id not in _MARKETS:
        _refresh_markets()
    return _MARKETS.get(token_id)


def _market_dict(m: Market) -> dict[str, Any]:
    return {
        "token_id": m.token_id, "condition_id": m.condition_id, "question": m.question,
        "outcome": m.outcome, "price": m.price, "volume_24h": m.volume_24h,
        "liquidity": m.liquidity, "spread": m.spread, "days_to_expiry": round(m.days_to_expiry, 2),
    }


def _decision_dict(d: TradeDecision) -> dict[str, Any]:
    return {
        "action": d.action, "p_calibrated": d.p_true, "raw_p_true": d.raw_p_true,
        "price": d.market_price, "edge": d.edge, "annualized_edge": d.annualized_edge,
        "days_to_expiry": d.days_to_expiry, "kelly_fraction": d.kelly_fraction,
        "size_usdc": d.size_usdc, "reasons": d.reasons,
    }


def _live_price_spread(token_id: str, fallback_price: float) -> tuple[float, float | None]:
    """Live book mid + spread_bps; falls back to the market snapshot price."""
    book = engine().client.fetch_order_book(token_id)
    if book is None:
        return fallback_price, None
    micro = compute_microstructure(book)
    return float(micro.get("mid") or fallback_price), micro.get("spread_bps")


# --- tools -------------------------------------------------------------------

@mcp.tool()
def scan_markets(limit: int = 20, min_volume_24h: float = 50000.0) -> list[dict]:
    """List active Polymarket markets (one row per YES/NO side), most-active first.

    Use this first to find candidates. Returns token_id/condition_id you pass to
    the other tools, plus price, 24h volume, liquidity and days to expiry.
    """
    _refresh_markets()
    out = [
        _market_dict(m) for m in _MARKETS.values()
        if 0.02 < m.price < 0.98 and m.volume_24h >= min_volume_24h
    ]
    out.sort(key=lambda r: r["volume_24h"], reverse=True)
    return out[:limit]


@mcp.tool()
def market_snapshot(token_id: str) -> dict:
    """Full Layer-1 data for one market side: price/volume/order-book microstructure/
    trade-flow reports plus the consolidated factor vector. No LLM — this is the
    evidence you reason over to estimate the true probability."""
    m = _get_market(token_id)
    if m is None:
        return {"error": f"market {token_id} not found among active markets"}
    state = engine().collect(m)
    return {
        "market": _market_dict(m),
        "price_report": state["price_report"],
        "volume_report": state["volume_report"],
        "orderbook_report": state["orderbook_report"],
        "trades_flow_report": state["trades_flow_report"],
        "factors": (state["raw"].get("features", {}) or {}).get("factors", {}),
    }


@mcp.tool()
def find_similar_markets(query: str, n: int = 3) -> list[dict]:
    """Retrieve semantically similar past markets (Chroma RAG), with resolved
    winner when known — useful context for judging a new market."""
    rag = engine().rag
    if rag is None:
        return []
    return [
        {"question": h["metadata"].get("question", h.get("document", "")),
         "resolved_winner": h["metadata"].get("resolved_winner")}
        for h in rag.query_similar(query, n=n)
    ]


@mcp.tool()
def size_position(p_true: float, token_id: str) -> dict:
    """Given YOUR estimated true probability that this side resolves YES, return
    the deterministic decision: edge vs the live price, fractional-Kelly size, and
    risk gates (liquidity / spread / 6% edge floor). This is the math, not an opinion."""
    m = _get_market(token_id)
    if m is None:
        return {"error": f"market {token_id} not found"}
    price, spread_bps = _live_price_spread(token_id, m.price)
    direction = "yes" if p_true >= price else "no"
    signal = Signal(direction=direction, p_true=p_true, conviction="medium", rationale="agent estimate")
    d = decide(signal, price, m.liquidity, spread_bps, engine().config, days_to_expiry=m.days_to_expiry)
    return _decision_dict(d)


@mcp.tool()
def paper_execute(token_id: str, side: str, size_usdc: float) -> dict:
    """Place a PAPER order (no real money) through the circuit breaker; updates the
    portfolio. ``side`` is 'buy' or 'sell' (sell exits the whole position)."""
    eng = engine()
    m = _get_market(token_id)
    if m is None:
        return {"error": f"market {token_id} not found"}
    ref_price, _ = _live_price_spread(token_id, m.price)
    book = eng.client.fetch_order_book(token_id)   # walk it for realistic slippage
    order = Order(token_id=token_id, side=side, size_usdc=size_usdc, ref_price=ref_price,
                  market=m.question, book=book)
    allowed, reason = eng.circuit_breaker.check(order, eng.portfolio)
    if not allowed:
        res = ExecutionResult("blocked", order, reason=reason)
    else:
        res = eng.execution_client.submit(order, eng.portfolio)
    return {"status": res.status, "reason": res.reason, "realized_pnl": res.realized_pnl,
            "portfolio": portfolio_status()}


@mcp.tool()
def portfolio_status() -> dict:
    """Current paper portfolio: cash, open positions, exposure, realised P&L."""
    p = engine().portfolio
    return {
        "cash": round(p.cash, 2),
        "exposure": round(p.exposure(), 2),
        "realized_pnl": round(p.realized_pnl(), 2),
        "open_positions": [
            {"token_id": pos.token_id, "market": pos.market, "shares": round(pos.shares, 1),
             "avg_price": round(pos.avg_price, 3)}
            for pos in p.positions.values()
        ],
    }


@mcp.tool()
def settle_markets() -> list[dict]:
    """Settle any held positions whose markets have resolved: book $1/$0 payout and
    realised P&L (no LLM reflection — the host agent reflects). Returns settled trades."""
    settled = engine().settle(reflect=False)
    return [
        {"question": s.get("question"), "side": s.get("side"), "won": s.get("won"),
         "realized_pnl": s.get("realized_pnl")}
        for s in settled
    ]


@mcp.tool()
def pnl_report() -> str:
    """Aggregate P&L / attribution over all logged trades (hit rate, realised P&L,
    avg return, decision mix)."""
    return engine().report()


@mcp.tool()
def evaluation_report() -> str:
    """Forecast-quality report: does our p_true actually beat the market price as a
    baseline? Brier / log-loss / calibration error (ECE) for the model vs the
    market, stratified by category. If we don't beat the market, the edge is noise."""
    return engine().evaluate()


def main() -> None:
    transport = "streamable-http" if "--http" in sys.argv else "stdio"
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
