"""End-to-end Layer 1 + Layer 2 graph with fake client and fake LLM."""
from __future__ import annotations

from polyagents.agents.schemas import Reflection, Signal, TradeDecision
from polyagents.dataflows.news import NewsClient
from polyagents.default_config import DEFAULT_CONFIG
from polyagents.graph.setup import build_analysis_graph
from polyagents.graph.state import build_initial_state


def test_full_pipeline_signal_decision_reflection(fake_client, fake_llm, sample_market):
    config = DEFAULT_CONFIG.copy()
    news = NewsClient(api_key=None)
    graph = build_analysis_graph(fake_client, news, config, fake_llm)

    final = graph.invoke(build_initial_state(sample_market, as_of="2026-06-08T00:00:00+00:00"))

    # Layer 1 still ran.
    assert final["features_report"]
    assert "features" in final["raw"]

    # Layer 2 produced all three structured outputs.
    assert isinstance(final["signal"], Signal)
    assert isinstance(final["trade_decision"], TradeDecision)
    assert isinstance(final["reflection"], Reflection)

    # The decision consumed the signal's p_true (0.70); raw is preserved, the
    # sized p_true is calibrated toward the market price.
    d = final["trade_decision"]
    assert d.raw_p_true == 0.70
    assert d.p_true != 0.70                    # shrunk toward market
    assert d.action in ("buy", "hold", "sell")
    assert final["signal_report"] and final["decision_report"] and final["reflection_report"]


def test_decision_buys_when_gates_pass(fake_client, fake_llm, sample_market):
    # sample price 0.45, fake signal p_true 0.70 -> edge +0.25. The fake book's
    # spread (~444bps) would trip the default 300bps gate, so widen it here to
    # exercise the buy path end to end.
    config = DEFAULT_CONFIG.copy()
    config["max_spread_bps"] = 1000.0
    graph = build_analysis_graph(fake_client, NewsClient(api_key=None), config, fake_llm)
    final = graph.invoke(build_initial_state(sample_market, as_of="2026-06-08T00:00:00+00:00"))
    assert final["trade_decision"].action == "buy"
    assert final["trade_decision"].size_usdc > 0


def test_decision_gated_by_wide_spread(fake_client, fake_llm, sample_market):
    # Default 300bps gate vs the fake book's ~444bps spread -> hold despite edge.
    config = DEFAULT_CONFIG.copy()
    graph = build_analysis_graph(fake_client, NewsClient(api_key=None), config, fake_llm)
    final = graph.invoke(build_initial_state(sample_market, as_of="2026-06-08T00:00:00+00:00"))
    d = final["trade_decision"]
    assert d.action == "hold"
    assert any("spread" in r for r in d.reasons)
