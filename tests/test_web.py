"""Tests for the web chat layer: tools, skills registry, endpoints (no LLM)."""
from __future__ import annotations


def test_build_tools_exposes_the_trading_surface():
    from polyagents.web.agent import build_tools

    names = {t.name for t in build_tools()}
    for expected in ("scan_markets", "market_snapshot", "size_position",
                     "paper_execute", "portfolio_status", "settle_markets",
                     "pnl_report", "evaluation_report",
                     "crypto_price", "crypto_24h", "crypto_klines",
                     "list_events", "recent_trades", "verify_trade_math"):
        assert expected in names


def test_skills_registry_lists_skill_folders():
    from polyagents.web.agent import list_skills

    skills = list_skills()
    ids = {s["id"] for s in skills}
    assert {"polymarket-trading", "market-research", "cross-market-arb"} <= ids
    pt = next(s for s in skills if s["id"] == "polymarket-trading")
    assert pt["name"] and pt["description"] and pt["body"]


def test_compose_prompt_selects_skills():
    from polyagents.web.agent import _compose_prompt

    only_trading = _compose_prompt(["polymarket-trading"])
    assert "p_true" in only_trading and "SKILL:" not in only_trading   # single skill, no header

    both = _compose_prompt(["polymarket-trading", "market-research"])
    assert "SKILL:" in both and "market research" in both.lower()

    # unknown selection falls back to all skills, never empty
    assert _compose_prompt(["nope"]).strip()


def test_mcp_registry_lists_servers_with_tools():
    from polyagents.web.agent import list_mcp_servers

    servers = list_mcp_servers()
    ids = {s["id"] for s in servers}
    assert {"polyagents", "crypto", "polydata", "compliance",
            "qlib-backtest", "polymarket-docs"} <= ids
    crypto = next(s for s in servers if s["id"] == "crypto")
    assert "crypto_price" in crypto["tools"] and crypto["in_chat"] is True
    polydata = next(s for s in servers if s["id"] == "polydata")
    assert polydata["in_chat"] is True
    docs = next(s for s in servers if s["id"] == "polymarket-docs")
    assert docs["in_chat"] is False


def test_server_app_has_routes():
    from polyagents.web.server import app

    paths = {r.path for r in app.routes}
    for p in ("/", "/api/chat", "/api/skills", "/api/mcp", "/api/portfolio",
              "/api/markets", "/api/backtest"):
        assert p in paths
