"""PolyAgentsGraph — the run entrypoint.

Wires up the read-only clients, compiles the graph once, and runs it per
market. Mirrors TradingAgents' ``TradingAgentsGraph`` shape (build once, run per
target).

  * ``collect(market)``  — Layer 1 only (deterministic data collection).
  * ``analyze(market)``  — Layer 1 + Layer 2 (signal → decision → reflection);
                           needs an LLM (``ANTHROPIC_API_KEY``), or inject one.

Quick read-only data smoke test against the most-active market:

    python -m polyagents
"""
from __future__ import annotations

from typing import Any

from polyagents.dataflows.forecaster import CandleForecaster, NullForecaster
from polyagents.dataflows.news import NewsClient
from polyagents.dataflows.polymarket_client import PolymarketDataClient
from polyagents.dataflows.sentiment import LexiconSentimentScorer, SentimentScorer
from polyagents.dataflows.types import Market
from polyagents.dataflows.utils import utcnow
from polyagents.default_config import DEFAULT_CONFIG
from polyagents.execution.agent import create_execution_agent
from polyagents.execution.circuit_breaker import CircuitBreaker
from polyagents.execution.clients import (
    ExecutionClient,
    LiveCLOBExecutionClient,
    PaperExecutionClient,
)
from polyagents.execution.portfolio import Portfolio
from polyagents.feedback.memory import MemoryStore, make_trade_record
from polyagents.feedback.reflection import reflect_on_outcome
from polyagents.feedback.report import pnl_report
from polyagents.feedback.settlement import resolve_winner, resolve_winning_token, settlement_pnl
from polyagents.rag.store import ChromaRAG
from polyagents.storage.db import DataStore

from .setup import build_analysis_graph, build_data_collection_graph, build_trading_graph
from .state import build_initial_state


class PolyAgentsGraph:
    def __init__(
        self,
        config: dict | None = None,
        scorer: SentimentScorer | None = None,
        forecaster: CandleForecaster | None = None,
        llm: Any | None = None,
        execution_client: ExecutionClient | None = None,
        store: DataStore | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG.copy()
        self.client = PolymarketDataClient.from_config(self.config)
        self.news_client = NewsClient(self.config.get("tavily_api_key"))
        # Layer 1 persistence (SQLite). On by default; disable via config.
        if store is not None:
            self.store = store
        elif self.config.get("persist_enabled", True):
            self.store = DataStore(self.config["db_path"])
        else:
            self.store = None
        # FinGPT / Kronos seams — swap these for model-backed implementations later.
        self.scorer = scorer or LexiconSentimentScorer()
        self.forecaster = forecaster or NullForecaster()
        self._llm = llm                 # lazily built on first analyze() if None
        self._data_graph = None
        self._analysis_graph = None
        self._trading_graph = None

        # Layer 3 — persistent across markets (held here, not in per-run state).
        self.portfolio = Portfolio(starting_cash=self.config["bankroll_usdc"])
        self.circuit_breaker = CircuitBreaker(self.config)
        self.execution_client = execution_client or self._default_execution_client()

        # Layer 4 — persistent decision log / memory (feeds lessons back in).
        self.memory = MemoryStore(self.config["memory_path"])
        # Polymarket/agents-style RAG: vectorise markets so the signal agent can
        # retrieve semantically similar past markets. Disabled if chromadb absent.
        self.rag = (
            ChromaRAG(path=self.config.get("chroma_path"))
            if self.config.get("rag_enabled", True) else None
        )

    def _default_execution_client(self) -> ExecutionClient:
        if self.config.get("execution_mode") == "live":
            return LiveCLOBExecutionClient(self.config)
        return PaperExecutionClient(slippage_bps=self.config["paper_slippage_bps"])

    # ----- graphs (compiled lazily) -----------------------------------------

    @property
    def data_graph(self):
        if self._data_graph is None:
            self._data_graph = build_data_collection_graph(
                self.client, self.news_client, self.config,
                scorer=self.scorer, forecaster=self.forecaster, store=self.store,
            )
        return self._data_graph

    def _get_llm(self):
        if self._llm is None:
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                model=self.config["anthropic_model"],
                temperature=self.config.get("anthropic_temperature", 0.0),
            )
        return self._llm

    @property
    def analysis_graph(self):
        if self._analysis_graph is None:
            self._analysis_graph = build_analysis_graph(
                self.client, self.news_client, self.config, self._get_llm(),
                scorer=self.scorer, forecaster=self.forecaster, memory=self.memory,
                store=self.store, rag=self.rag,
            )
        return self._analysis_graph

    @property
    def trading_graph(self):
        if self._trading_graph is None:
            execute_node = create_execution_agent(
                self.execution_client, self.portfolio, self.circuit_breaker,
                data_client=self.client,
            )
            self._trading_graph = build_trading_graph(
                self.client, self.news_client, self.config, self._get_llm(), execute_node,
                scorer=self.scorer, forecaster=self.forecaster, memory=self.memory,
                store=self.store, rag=self.rag,
            )
        return self._trading_graph

    # ----- runs --------------------------------------------------------------

    def _persist_market(self, market: Market) -> None:
        if self.store is not None:
            self.store.record_market(market)
        if self.rag is not None:
            self.rag.index_market(market)   # vectorise for similar-market retrieval

    def collect(self, market: Market, as_of: str | None = None) -> dict[str, Any]:
        """Layer 1 only: data collection for one market; returns final state."""
        as_of = as_of or utcnow().isoformat()
        self._persist_market(market)
        return self.data_graph.invoke(build_initial_state(market, as_of))

    def analyze(self, market: Market, as_of: str | None = None) -> dict[str, Any]:
        """Layer 1 + Layer 2: collect, then signal → decision → reflection."""
        as_of = as_of or utcnow().isoformat()
        self._persist_market(market)
        return self.analysis_graph.invoke(build_initial_state(market, as_of))

    def trade(self, market: Market, as_of: str | None = None) -> dict[str, Any]:
        """Layer 1 + 2 + 3: analyse then execute (paper by default) through the
        circuit breaker; the portfolio persists across calls. The decision is
        logged to memory (Layer 4) as a pending record."""
        as_of = as_of or utcnow().isoformat()
        self._persist_market(market)
        state = self.trading_graph.invoke(build_initial_state(market, as_of))
        self.memory.record(make_trade_record(state))
        return state

    # ----- Layer 4: feedback loop -------------------------------------------

    def settle(self, reflect: bool = True) -> list[dict]:
        """Resolve pending trades whose markets have closed: settle the paper
        position, book realised P&L, and (when ``reflect``) write a reflection
        lesson to memory. Set ``reflect=False`` to skip the LLM (e.g. when the
        MCP host agent does its own reflection — no internal key needed).
        Returns the records that were settled this pass."""
        settled: list[dict] = []
        for rec in self.memory.pending():
            market_raw = self.client.fetch_market_by_condition(rec["condition_id"])
            win_token = resolve_winning_token(market_raw or {})
            if win_token is None:
                continue                      # not resolved yet
            won = rec["token_id"] == win_token        # robust: compare token, not label
            winner = resolve_winner(market_raw or {}) # best-effort label for display
            if self.rag is not None:
                self.rag.annotate_outcome(rec["condition_id"], winner)  # close the RAG loop
            pos = self.portfolio.positions.get(rec["token_id"])
            pnl = ret = None
            if pos is not None:               # we actually hold it -> settle paper payout
                cost = pos.cost_basis
                self.portfolio.apply_sell_close(rec["token_id"], 1.0 if won else 0.0, utcnow())
                pnl = settlement_pnl(won, pos.shares, pos.avg_price)
                ret = (pnl / cost) if cost else None
            updates = {
                "status": "resolved", "resolved_winner": winner, "won": won,
                "realized_pnl": pnl, "realized_return": ret,
            }
            rec.update(updates)
            updates["lesson"] = None
            if reflect:
                try:
                    lesson = reflect_on_outcome(self._get_llm(), rec)
                    updates["lesson"] = f"{lesson.summary} Next time: {lesson.what_to_change}"
                except Exception:
                    updates["lesson"] = None
            self.memory.update(rec["record_id"], **updates)
            settled.append({**rec, **updates})
        return settled

    def report(self) -> str:
        """Aggregate P&L / attribution over the decision log."""
        return pnl_report(self.memory.all())

    def evaluate(self) -> str:
        """Calibration / skill report: does p_true beat the market baseline?
        (Brier / log-loss / ECE vs the market price, stratified by category.)"""
        from polyagents.evaluation.evaluate import evaluate as _eval, format_report
        return format_report(_eval(self.memory.all()))

    def most_active_market(self) -> Market | None:
        """Discovery helper: the single most-active tradeable side right now."""
        raw = self.client.list_active_markets(limit=self.config["markets_limit"])
        markets = self.client.to_markets(raw)
        return markets[0] if markets else None

    def close(self) -> None:
        self.client.close()
        if self.store is not None:
            self.store.close()


def _format_state(state: dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        state["market_context"],
        "=" * 70,
        f"\n[price]\n{state['price_report']}",
        f"\n[volume]\n{state['volume_report']}",
        f"\n[orderbook]\n{state['orderbook_report']}",
        f"\n[trades_flow]\n{state['trades_flow_report']}",
        f"\n[news]\n{state['news_report']}",
        f"\n[features]\n{state['features_report']}",
    ]
    # Layer 2/3 sections appear only when analyze()/trade() ran.
    for key, label in (("signal_report", "signal"), ("decision_report", "decision"),
                       ("reflection_report", "reflection"), ("execution_report", "execution")):
        if state.get(key):
            lines.append(f"\n[{label}]\n{state[key]}")
    return "\n".join(lines)


def main() -> None:
    ta = PolyAgentsGraph()
    try:
        market = ta.most_active_market()
        if market is None:
            print("No active markets returned by Gamma.")
            return
        state = ta.collect(market)
        print(_format_state(state))
    finally:
        ta.close()


if __name__ == "__main__":
    main()
