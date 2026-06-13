"""Execution agent — the final graph node.

Turns the Layer 2 :class:`TradeDecision` into an order, runs it through the
circuit breaker, and submits to the execution client. The portfolio and breaker
live on the orchestrator (persistent across markets), so the node closes over
them rather than reading them from per-run state.
"""
from __future__ import annotations

from typing import Any, Callable

from .circuit_breaker import CircuitBreaker
from .clients import ExecutionClient
from .portfolio import Portfolio
from .types import ExecutionResult, Order

Node = Callable[[dict], dict]


def _ref_price(state: dict, side: str) -> float:
    ob = (state.get("raw", {}) or {}).get("orderbook", {}) or {}
    want = ob.get("best_ask") if side == "buy" else ob.get("best_bid")
    return float(want or ob.get("mid") or state.get("market_price") or 0.0)


def _format(result: ExecutionResult, portfolio: Portfolio) -> str:
    head = f"EXECUTION: {result.status.upper()}"
    if result.reason:
        head += f" — {result.reason}"
    tail = (
        f"\nPortfolio: cash ${portfolio.cash:,.2f}, "
        f"{len(portfolio.positions)} open, exposure ${portfolio.exposure():,.2f}, "
        f"realised P&L ${portfolio.realized_pnl():+,.2f}"
    )
    return head + tail


def create_execution_agent(
    client: ExecutionClient,
    portfolio: Portfolio,
    breaker: CircuitBreaker,
    data_client=None,
) -> Node:
    """``data_client`` (a PolymarketDataClient) is used to fetch the live order
    book so paper fills walk it for realistic slippage."""

    def node(state: dict) -> dict[str, Any]:
        decision = state["trade_decision"]
        if decision.action == "hold":
            res = ExecutionResult("skipped", reason="decision was HOLD")
            return {"execution_result": res, "execution_report": _format(res, portfolio)}

        book = None
        if data_client is not None:
            book = data_client.fetch_order_book(state["token_id"])
        order = Order(
            token_id=state["token_id"],
            side=decision.action,
            size_usdc=decision.size_usdc,
            ref_price=_ref_price(state, decision.action),
            market=state.get("question", ""),
            book=book,
        )
        allowed, reason = breaker.check(order, portfolio)
        if not allowed:
            res = ExecutionResult("blocked", order, reason=reason)
        else:
            res = client.submit(order, portfolio)
        return {"execution_result": res, "execution_report": _format(res, portfolio)}

    return node
