"""Tests for the ChromaDB RAG (Polymarket/agents-style market retrieval)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polyagents.dataflows.types import Market
from polyagents.rag.store import ChromaRAG


def _mkt(cid, question, price=0.5, outcome="YES") -> Market:
    return Market(
        market_id=cid, condition_id=cid, question=question, description="",
        outcome=outcome, token_id=f"tok_{cid}", price=price, volume_24h=1.0,
        liquidity=1.0, spread=0.01, days_to_expiry=3.0,
        expiry=datetime.now(timezone.utc) + timedelta(days=3),
    )


def test_index_and_query_similar_semantic():
    rag = ChromaRAG()                       # ephemeral, local embeddings
    assert rag.enabled
    rag.index_market(_mkt("c1", "Will Bitcoin hit $100k by 2026?"))
    rag.index_market(_mkt("c2", "Will Ethereum flip Bitcoin in market cap?"))
    rag.index_market(_mkt("c3", "Will the Lakers win the NBA finals?"))
    assert rag.count() == 3

    hits = rag.query_similar("crypto BTC price prediction", n=2)
    qs = [h["metadata"]["question"] for h in hits]
    assert any("Bitcoin" in q for q in qs)
    assert "Will the Lakers win the NBA finals?" not in qs   # sports is dissimilar


def test_query_excludes_self():
    rag = ChromaRAG()
    rag.index_market(_mkt("self", "Will Bitcoin hit $100k?"))
    rag.index_market(_mkt("other", "Will Bitcoin crash below $20k?"))
    hits = rag.query_similar("Bitcoin price", n=5, exclude_id="self")
    assert all(h["id"] != "self" for h in hits)


def test_annotate_outcome_round_trips():
    rag = ChromaRAG()
    rag.index_market(_mkt("c1", "Will it rain tomorrow?"))
    rag.annotate_outcome("c1", "NO")
    hit = rag.query_similar("rain weather", n=1)[0]
    assert hit["metadata"].get("resolved_winner") == "NO"


def test_graceful_when_chroma_unavailable():
    class _BoomClient:
        def get_or_create_collection(self, name):
            raise RuntimeError("no chroma")

    rag = ChromaRAG(client=_BoomClient())
    assert rag.enabled is False
    rag.index_market(_mkt("c1", "Q?"))      # no-op, no raise
    assert rag.query_similar("anything") == []
    assert rag.count() == 0


def test_signal_agent_injects_similar_markets():
    from polyagents.agents.schemas import Signal
    from polyagents.agents.signal_agent import create_signal_agent

    rag = ChromaRAG()
    rag.index_market(_mkt("past", "Will Bitcoin hit $100k by 2026?"))

    captured: dict = {}

    class SpyStructured:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return Signal(direction="yes", p_true=0.6, conviction="low", rationale="r")

    class SpyLLM:
        def with_structured_output(self, schema):
            return SpyStructured()

    node = create_signal_agent(SpyLLM(), rag=rag)
    node({"question": "Will Bitcoin reach 100000 dollars?", "condition_id": "current",
          "market_price": 0.5, "raw": {}})
    assert "similar past markets" in captured["prompt"].lower()
    assert "Bitcoin" in captured["prompt"]
