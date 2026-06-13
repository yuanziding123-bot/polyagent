"""The shared state — polyagents' blackboard.

Like TradingAgents' ``AgentState``, a single ``MarketState`` flows through every
node. Nodes read the fields they need and return a partial dict that LangGraph
merges back in. Extending ``MessagesState`` gives us a reducer-backed
``messages`` list for free, so the later LLM analyst layer can drop in without a
state change.

The data-collection layer fills the ``*_report`` strings (for humans / LLMs) and
the ``raw`` dict (structured numbers for detectors / sizing).
"""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import MessagesState

from polyagents.dataflows.interface import get_market_context
from polyagents.dataflows.types import Market


class MarketState(MessagesState):
    # --- target identity (resolved once at run start) ---
    market_id: Annotated[str, "Gamma market id under analysis"]
    condition_id: Annotated[str, "On-chain condition id"]
    token_id: Annotated[str, "CLOB token id for the analysed outcome side"]
    question: Annotated[str, "Market question"]
    outcome: Annotated[str, "Analysed side: YES or NO"]
    market_price: Annotated[float, "Last market price for the analysed side"]
    liquidity: Annotated[float, "Market liquidity (USDC) — risk gate input"]
    volume_24h: Annotated[float, "24h volume (USDC)"]
    days_to_expiry: Annotated[float, "Days to resolution — time-annualised edge gate"]
    as_of: Annotated[str, "ISO timestamp the collection run was anchored to"]
    market_context: Annotated[str, "Deterministic market identity block"]

    # --- Layer 1: data-collection reports (filled by collector nodes) ---
    price_report: Annotated[str, "Price-history summary"]
    volume_report: Annotated[str, "Reconstructed-volume summary"]
    orderbook_report: Annotated[str, "L2 microstructure summary (depth, micro-price, pressure)"]
    trades_flow_report: Annotated[str, "Buy/sell flow-imbalance summary"]
    news_report: Annotated[str, "Relevant news summary with sentiment"]
    features_report: Annotated[str, "Consolidated factor vector summary"]

    # --- structured numbers, keyed by source (e.g. raw["price"], raw["orderbook"]) ---
    raw: Annotated[dict[str, Any], "Structured numeric outputs from each collector"]

    # --- Layer 2: decision engine (signal -> decision -> reflection) ---
    signal: Annotated[Any, "Signal agent output (Signal)"]
    signal_report: Annotated[str, "Signal summary"]
    trade_decision: Annotated[Any, "Decision agent output (TradeDecision)"]
    decision_report: Annotated[str, "Decision summary"]
    reflection: Annotated[Any, "Reflection agent output (Reflection)"]
    reflection_report: Annotated[str, "Reflection summary"]

    # --- Layer 3: execution ---
    execution_result: Annotated[Any, "ExecutionResult from the execution agent"]
    execution_report: Annotated[str, "Execution + portfolio summary"]


def build_initial_state(market: Market, as_of: str) -> dict[str, Any]:
    """Seed a fresh ``MarketState`` from a resolved :class:`Market`."""
    return {
        "messages": [],
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "token_id": market.token_id,
        "question": market.question,
        "outcome": market.outcome,
        "market_price": market.price,
        "liquidity": market.liquidity,
        "volume_24h": market.volume_24h,
        "days_to_expiry": market.days_to_expiry,
        "as_of": as_of,
        "market_context": get_market_context(market),
        "price_report": "",
        "volume_report": "",
        "orderbook_report": "",
        "trades_flow_report": "",
        "news_report": "",
        "features_report": "",
        "raw": {},
    }
