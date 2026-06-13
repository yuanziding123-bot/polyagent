"""Tests for the paper execution client and the execution agent node."""
from __future__ import annotations

from pytest import approx

from polyagents.agents.schemas import TradeDecision
from polyagents.execution.agent import create_execution_agent
from polyagents.execution.circuit_breaker import CircuitBreaker
from polyagents.execution.clients import PaperExecutionClient
from polyagents.execution.portfolio import Portfolio
from polyagents.execution.types import Order, Position
from polyagents.dataflows.types import OrderBook, OrderBookLevel
from polyagents.default_config import DEFAULT_CONFIG


def test_paper_buy_fills_with_slippage_and_books():
    pf = Portfolio(500.0)
    client = PaperExecutionClient(slippage_bps=100.0)   # 1%
    order = Order("t", "buy", size_usdc=50.0, ref_price=0.50)
    res = client.submit(order, pf)
    assert res.status == "filled"
    assert res.fill.price == 0.505                       # 0.50 * 1.01
    assert pf.cash == 450.0
    assert pf.positions["t"].shares == 50.0 / 0.505


def test_paper_buy_walks_the_book_for_slippage():
    pf = Portfolio(500.0)
    client = PaperExecutionClient(slippage_bps=0.0)
    # $5 at 0.50, then deeper at 0.60 — a $20 buy must walk up
    book = OrderBook("t", bids=[OrderBookLevel(0.49, 100)],
                     asks=[OrderBookLevel(0.50, 10), OrderBookLevel(0.60, 100)])
    res = client.submit(Order("t", "buy", size_usdc=20.0, ref_price=0.50, book=book), pf)
    assert res.status == "filled"
    assert res.fill.price > 0.50               # avg fill worse than the touch (impact)
    assert res.fill.notional == approx(20.0)   # full notional deployed


def test_paper_buy_partial_when_book_thin():
    pf = Portfolio(500.0)
    client = PaperExecutionClient(slippage_bps=0.0)
    book = OrderBook("t", bids=[], asks=[OrderBookLevel(0.50, 10)])   # only $5 of asks
    res = client.submit(Order("t", "buy", size_usdc=20.0, ref_price=0.50, book=book), pf)
    assert res.status == "filled"
    assert res.fill.notional == approx(5.0)    # only filled what the book had
    assert "partial" in res.reason


def test_paper_sell_closes_and_realizes():
    pf = Portfolio(500.0)
    client = PaperExecutionClient(slippage_bps=0.0)
    client.submit(Order("t", "buy", 50.0, 0.50), pf)
    res = client.submit(Order("t", "sell", 0.0, 0.60), pf)
    assert res.status == "filled"
    assert res.realized_pnl == approx(10.0)              # (0.6-0.5)*100
    assert "t" not in pf.positions


def _node_state(decision):
    return {
        "token_id": "t", "question": "Q?", "market_price": 0.50,
        "trade_decision": decision,
        "raw": {"orderbook": {"best_bid": 0.49, "best_ask": 0.51, "mid": 0.50}},
    }


def _node(pf):
    return create_execution_agent(PaperExecutionClient(slippage_bps=0.0), pf, CircuitBreaker(DEFAULT_CONFIG.copy()))


def test_execution_node_fills_a_buy():
    pf = Portfolio(500.0)
    out = _node(pf)(_node_state(TradeDecision("buy", 0.70, 0.50, 0.20, 0.05, 25.0, ["r"])))
    assert out["execution_result"].status == "filled"
    assert "t" in pf.positions
    assert "FILLED" in out["execution_report"]


def test_execution_node_skips_hold():
    pf = Portfolio(500.0)
    out = _node(pf)(_node_state(TradeDecision("hold", 0.50, 0.50, 0.0, 0.0, 0.0, ["thin"])))
    assert out["execution_result"].status == "skipped"
    assert not pf.positions


def test_execution_node_blocks_on_breaker():
    pf = Portfolio(10.0)   # not enough cash for a $25 buy -> breaker blocks
    out = _node(pf)(_node_state(TradeDecision("buy", 0.70, 0.50, 0.20, 0.05, 25.0, ["r"])))
    assert out["execution_result"].status == "blocked"
    assert not pf.positions
