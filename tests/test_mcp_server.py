"""Tests for the polyagents MCP server tools (injected fake engine, no network)."""
from __future__ import annotations

from pytest import approx

from polyagents import mcp_server
from polyagents.default_config import DEFAULT_CONFIG
from polyagents.execution.circuit_breaker import CircuitBreaker
from polyagents.execution.clients import PaperExecutionClient
from polyagents.execution.portfolio import Portfolio

from .conftest import TOKEN


class _Client:
    def __init__(self, market, book):
        self._m = market
        self._book = book

    def list_active_markets(self, limit):
        return [{}]

    def to_markets(self, raw):
        return [self._m]

    def fetch_order_book(self, token_id):
        return self._book


class _Engine:
    def __init__(self, market, book, config):
        self.config = config
        self.client = _Client(market, book)
        self.portfolio = Portfolio(config["bankroll_usdc"])
        self.circuit_breaker = CircuitBreaker(config)
        self.execution_client = PaperExecutionClient(slippage_bps=0.0)
        self.rag = None


def _setup(fake_client, sample_market, max_spread=1000.0):
    cfg = DEFAULT_CONFIG.copy()
    cfg["max_spread_bps"] = max_spread
    eng = _Engine(sample_market, fake_client.fetch_order_book(TOKEN), cfg)
    mcp_server.set_engine(eng)
    return eng


def test_scan_markets(fake_client, sample_market):
    _setup(fake_client, sample_market)
    rows = mcp_server.scan_markets(limit=5, min_volume_24h=1000)
    assert any(r["token_id"] == TOKEN for r in rows)
    assert rows[0]["question"] == sample_market.question


def test_size_position_buys_on_edge(fake_client, sample_market):
    _setup(fake_client, sample_market)         # mid 0.45, spread gate relaxed
    d = mcp_server.size_position(p_true=0.70, token_id=TOKEN)
    assert d["action"] == "buy"
    assert d["raw_p_true"] == 0.70
    assert d["p_calibrated"] < 0.70            # shrunk toward market price
    assert 0 < d["edge"] < 0.25                # calibrated edge is smaller than raw
    assert d["annualized_edge"] > 0
    assert d["size_usdc"] > 0


def test_size_position_holds_on_wide_spread(fake_client, sample_market):
    _setup(fake_client, sample_market, max_spread=300.0)   # fake book ~444bps trips gate
    d = mcp_server.size_position(p_true=0.70, token_id=TOKEN)
    assert d["action"] == "hold"
    assert any("spread" in r for r in d["reasons"])


def test_paper_execute_and_portfolio(fake_client, sample_market):
    eng = _setup(fake_client, sample_market)
    res = mcp_server.paper_execute(TOKEN, "buy", 25.0)
    assert res["status"] == "filled"
    assert len(eng.portfolio.positions) == 1
    status = mcp_server.portfolio_status()
    assert status["cash"] == approx(475.0)     # 500 - 25
    assert status["open_positions"][0]["token_id"] == TOKEN


def test_paper_execute_blocked_by_breaker(fake_client, sample_market):
    eng = _setup(fake_client, sample_market)
    eng.portfolio.cash = 5.0                    # not enough for a $25 buy
    res = mcp_server.paper_execute(TOKEN, "buy", 25.0)
    assert res["status"] == "blocked"
    assert not eng.portfolio.positions


def test_tools_are_registered():
    # FastMCP exposes the registered tool names
    names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
    for expected in ("scan_markets", "market_snapshot", "size_position",
                     "paper_execute", "portfolio_status", "settle_markets", "pnl_report"):
        assert expected in names
