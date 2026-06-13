# polyagents

A LangGraph multi-agent framework for **Polymarket** prediction markets, built
from scratch. The architecture mirrors
[TradingAgents](../TradingAgents) — a shared state ("blackboard") flows through
graph nodes, each node reads it, does its job, and writes a partial update back
— while the data logic is adapted from the proven
[polymarket](../) reference implementation.

The project is built layer by layer. **Layer 1** (data collection) gathers
everything about one market into a typed state; **Layer 2** (decision engine)
turns that into a sized, risk-gated trade; **Layer 3** (execution) fills it on a
paper (or live CLOB) venue through a circuit breaker; **Layer 4** (feedback)
settles resolved trades, reflects on the realised result, and feeds the lesson
back into future decisions.

```
   ── L1: data collection ──┐  ┌── L2: decision engine ──┐  ┌─ L3: execution ─
START ► … ► features ► signal ► decision ► reflection ► execute ► END
            factor      LLM      Kelly +     LLM self-     paper/live +
            vector      p_true   risk gate   critique      circuit breaker
                          ▲                                     │
            L4 feedback   └──── lessons ◄── reflect ◄── settle ─┘  (after resolution)
```

`collect(market)` runs Layer 1 only (no LLM/keys); `analyze(market)` adds
Layer 2 (needs an Anthropic key, or inject an `llm`); `trade(market)` adds
Layer 3 execution (paper by default) and logs the decision; `settle()` runs
Layer 4 once markets resolve.

### Layer 1 capabilities (tracking the Merakku v3.0 Layer 1 projects)

| Source project | What we built | Module |
|---|---|---|
| **Polymarket py-clob-client** (P0) | Order book read via the **official CLOB SDK** (richer L2 depth), public REST `/book` as fallback | `dataflows/polymarket_client.py` |
| **MarketLens** (P0) | (a) native L2 microstructure on the live book — micro-price, depth imbalance, book pressure, spread (bps), queue-at-touch; (b) **MarketLens SDK adapter** for tick-level *historical* L2 (resolved markets / backtesting) — needs `MARKETLENS_API_KEY` | `dataflows/microstructure.py`, `dataflows/marketlens_client.py` |
| **FinGPT** (P0) | Sentiment scoring on news; `SentimentScorer` protocol + deterministic lexicon default, LLM/FinGPT pluggable | `dataflows/sentiment.py` |
| **Alpha DevBox** (P0) | Deterministic factor extraction — joins every collector's output into one named factor vector | `dataflows/features.py` |
| **Kronos** (P3) | `CandleForecaster` protocol seam over the close series; `NullForecaster` default | `dataflows/forecaster.py` |
| **Polyseer / poly_data** (P1) | planned — real-time market intelligence & event/historical retrieval | — |
| **pmxt / FinceptTerminal** (P2) | reference only — no code | — |

The FinGPT and Kronos seams are **injectable**: `PolyAgentsGraph(scorer=..., forecaster=...)`
swaps the lightweight built-ins for model-backed implementations without touching the graph.
The order book uses the official SDK by default (no keys needed for public L1 reads);
set `use_clob_sdk: False` in config to force the REST path.

#### Persistence (SQLite)

Layer 1 writes through to a SQLite store (`storage/db.py`, stdlib only) at
`~/.polyagents/cache/polyagents.db`: `markets`, `candles`, `trades`,
`orderbook_snapshots`, and `collections` (the full factor bundle per run). Two
wins: **caching** — the volume reconstruction reuses cached trades (tracked by a
fetch watermark), so re-collecting a market doesn't re-paginate a week of
`/trades`; and **history** — candles / books / collections accumulate for later
ML / backtesting. On by default; `persist_enabled: False` runs fully in-memory.

```python
ta = PolyAgentsGraph()
ta.collect(market)
print(ta.store.counts())   # {'markets': 1, 'candles': 169, 'trades': 3418, ...}
```

### Layer 2 — decision engine (Merakku v3.0 three-agent architecture)

| Agent | Role | How | Module |
|---|---|---|---|
| **Signal** | factors + flow + sentiment → estimated true probability (`p_true`, direction, conviction) | LLM (Claude), structured output | `agents/signal_agent.py` |
| **Decision** | edge vs price → fractional-Kelly size + hard risk gates (liquidity, spread, edge floor) | **deterministic** (risk embedded, auditable) | `agents/decision_agent.py`, `agents/risk.py` |
| **Reflection** | pre-trade self-critique: risk flags, shaky assumptions, OOD | LLM (Claude), structured output | `agents/reflection_agent.py` |

The decision agent is intentionally **not** an LLM — sizing and risk are math.
Two safeguards beyond raw edge (from review feedback): the model `p_true` is
first **calibrated** toward the market price (a calibrated baseline; the LLM is
noisy — `agents/calibration.py`), and entries are gated on the **time-annualised**
return / APY, not just a flat 6% edge (a 6% edge in 9 days ≠ 9 months). Then:
`edge = p_cal − price`, `f* = (q−p)/(1−p)`, quarter-Kelly capped at 5% of bankroll.
Paper fills **walk the order book** for realistic slippage (filling at mid would
overstate P&L and poison the feedback loop). The
`llm` is injectable: `PolyAgentsGraph(llm=...)`, and tests use a fake LLM so the
whole pipeline runs without a key or network.

**RAG over markets (Polymarket/agents-style).** The official Polymarket/agents
framework (archived, OpenAI-based, Python 3.9) overlaps what we already have
except its **Chroma RAG**. We adopt that natively in `rag/store.py` with
`chromadb` (local all-MiniLM embeddings — free, no key): every collected market
is vectorised, and the signal agent retrieves semantically *similar past
markets* (with their resolved winner when known) as extra context. Disabled
gracefully if `chromadb` isn't installed (`rag_enabled: False`).

```python
from polyagents.graph.orchestrator import PolyAgentsGraph
ta = PolyAgentsGraph()                 # needs ANTHROPIC_API_KEY for analyze()
market = ta.most_active_market()
state = ta.analyze(market)             # signal -> decision -> reflection
print(state["decision_report"])
```

### Layer 3 — execution (Merakku v3.0, NautilusTrader-inspired)

The decision becomes an order, passes a circuit breaker, and fills on a venue —
NautilusTrader's port/adapter shape, scaled to Polymarket:

| Piece | Role | Module |
|---|---|---|
| `ExecutionClient` (port) | venue interface; strategy depends only on this | `execution/clients.py` |
| `PaperExecutionClient` | **default**: simulate fills at touch ± slippage, book to portfolio | `execution/clients.py` |
| `LiveCLOBExecutionClient` | real GTC limit orders via official SDK (needs `POLYMARKET_PRIVATE_KEY`); **gated, never default** | `execution/clients.py` |
| `Portfolio` | virtual cash, positions, realised/unrealised P&L | `execution/portfolio.py` |
| `CircuitBreaker` | pre-trade gates: daily-loss halt, exposure cap, max concurrent, consecutive-loss cooldown, cash | `execution/circuit_breaker.py` |

```python
ta = PolyAgentsGraph()                 # execution_mode="paper" by default
state = ta.trade(market)               # … reflection -> execute (circuit-breaker gated)
print(state["execution_report"])       # FILLED / BLOCKED / SKIPPED + portfolio
print(ta.portfolio.cash, ta.portfolio.positions, ta.portfolio.realized_pnl())
```

The portfolio + breaker persist on the `PolyAgentsGraph` across markets. Set
`execution_mode: "live"` (and `POLYMARKET_PRIVATE_KEY`) to place real orders —
off by default so nothing trades for real unless you opt in.

### Layer 4 — feedback loop (Merakku v3.0; TradingAgents-style memory)

The learning half: every `trade()` is logged; once a market resolves, `settle()`
books realised P&L and reflects on it; lessons are injected into future signals.

| Piece | Role | Module |
|---|---|---|
| `MemoryStore` | persistent decision log (JSONL at `~/.polyagents/memory/`) | `feedback/memory.py` |
| settlement | resolve winner from Gamma **by token id** (label-agnostic), pay paper $1/$0 | `feedback/settlement.py` |
| `reflect_on_outcome` | LLM attribution → a `Lesson` (what worked / what to change) | `feedback/reflection.py` |
| lesson injection | recent lessons (same-market first) prepended to the signal prompt | `agents/signal_agent.py` |
| `pnl_report` | hit rate, realised P&L, avg return, decision mix | `feedback/report.py` |

```python
ta = PolyAgentsGraph()
ta.trade(market)            # logs a pending record
# … later, after markets resolve …
ta.settle()                 # books P&L + writes a reflection lesson per resolved trade
print(ta.report())          # hit rate / realised P&L / attribution
```

Settlement keys on the **CLOB token id**, not a YES/NO label, so markets with
custom outcomes (player/candidate names) settle correctly. Portfolio-optimisation
seams (Riskfolio-Lib / skfolio) and full Langfuse tracing remain future work.

### Polymarket docs MCP

The official [Polymarket documentation MCP](https://docs.polymarket.com/mcp) (a
docs **search/read** server — not a market-data feed) is wired in two ways:

- **Dev-time** — [`.mcp.json`](.mcp.json) registers it with Claude Code so the
  coding agent can look up Polymarket API/contract details while building polyagents.
- **Run-time** — `polyagents/mcp_tools.py` turns it into LangGraph tools via
  `langchain-mcp-adapters`, for the later decision-layer agents:

  ```python
  from polyagents.mcp_tools import load_mcp_tools_sync
  tools = load_mcp_tools_sync()   # [search_polymarket_documentation, query_docs_filesystem...]
  ```

  Servers are configured under `mcp_servers` in `default_config.py`; an empty map
  short-circuits with no network call and no extra imports.

## Platform integration — polyagents as skills + MCP

polyagents is also packaged as **skills + an MCP server** so it can plug into an
Alpha DevBox-style chat platform: the platform hosts a Claude agent, the agent
loads the **skill** and connects to the **MCP server**, and the user trades
through chat. The platform is the shell; polyagents provides the capabilities.

```
chat (Alpha DevBox)  →  agent (Claude)  →  skills/*  +  polyagents MCP tools
```

- **MCP server** — `polyagents/mcp_server.py` (`FastMCP`) exposes the engine as
  deterministic, JSON-returning tools: `scan_markets`, `market_snapshot`,
  `find_similar_markets`, `size_position` (calibration + Kelly + time-annualised
  gates), `paper_execute` (walk-the-book fills, circuit-breaker gated),
  `portfolio_status`, `settle_markets`, `pnl_report`, `evaluation_report`.
  The **host agent does the reasoning**; the tools need no internal LLM/key.

  ```bash
  python -m polyagents.mcp_server         # stdio (Claude / Alpha DevBox)
  python -m polyagents.mcp_server --http   # streamable-http on :8000
  ```

- **Skills** — `skills/<name>/SKILL.md` teaches the agent the workflow (scan →
  snapshot → estimate p_true → size → paper-trade → settle/review) and the
  discipline. Ships with `skills/polymarket-trading/`. Adding a skill = expose a
  few more `@mcp.tool()`s + write one `SKILL.md` — see `skills/README.md`.
  Everything is paper / read-only by default.

## Layout

```
polyagents/
  default_config.py        # config dict + env overrides (mirrors TA default_config)
  dataflows/               # the data interface — "tools" the graph calls
    polymarket_client.py   # Gamma + data-api over httpx; order book via official py-clob-client SDK
    news.py                # Tavily news search (graceful no-key fallback)
    volume.py              # rebuild candle volume from /trades
    microstructure.py      # MarketLens-inspired L2 features (on the live book)
    marketlens_client.py   # MarketLens SDK adapter — historical L2 for resolved markets (needs key)
    sentiment.py           # FinGPT-inspired sentiment scorer (pluggable)
    forecaster.py          # Kronos-inspired CandleForecaster seam
    features.py            # Alpha DevBox-inspired factor join
    interface.py           # high-level fetch+format functions (report + structured data)
    types.py               # Market / Candle / OrderBook domain types
  agents/                  # Layer 2 — decision engine
    schemas.py             # Signal / Reflection (pydantic) + TradeDecision
    signal_agent.py        # LLM: estimate true probability
    decision_agent.py      # deterministic: edge + Kelly + risk gates
    risk.py                # pure risk math (edge, Kelly fraction, effective price)
    reflection_agent.py    # LLM: pre-trade self-critique
  execution/               # Layer 3 — execution
    types.py               # Order / Fill / Position / ExecutionResult
    portfolio.py           # virtual cash, positions, realised P&L
    circuit_breaker.py     # pre-trade safety gates
    clients.py             # ExecutionClient port + Paper / LiveCLOB adapters
    agent.py               # execution node (decision -> breaker -> venue)
  feedback/                # Layer 4 — feedback loop
    memory.py              # persistent decision log + lesson injection source
    settlement.py          # resolve winner (by token) + paper payout
    reflection.py          # LLM outcome reflection -> Lesson
    report.py              # P&L / attribution report
  evaluation/              # forecast-quality / calibration (peer to L1-L4)
    metrics.py             # Brier / log-loss / ECE + calibration curve
    evaluate.py            # model vs MARKET baseline, stratified by category
  agents/calibration.py    # shrink p_true toward market (Calibrator)
  storage/
    db.py                  # SQLite store: markets/candles/trades/orderbook/collections + trades cache
  rag/
    store.py               # ChromaDB RAG over markets (Polymarket/agents-style retrieval)
  mcp_tools.py             # load configured MCP servers (Polymarket docs) as LangGraph tools
  mcp_server.py            # FastMCP server: expose the engine as tools for a host platform
  graph/
    state.py               # MarketState TypedDict (L1+L2 fields) + initial-state builder
    data_collection.py     # collector node factories (incl. features join)
    setup.py               # build_data_collection_graph / _analysis_graph / _trading_graph
    orchestrator.py        # PolyAgentsGraph — collect() / analyze() / trade()
skills/
  README.md                # how to add skills (expose @mcp.tool + write SKILL.md)
  polymarket-trading/
    SKILL.md               # the trading workflow + discipline for the host agent
```

## Quick start

```powershell
# Uses the workspace venv (already provisioned at C:\polymarket\.venv)
C:\polymarket\.venv\Scripts\python.exe -m pip install -r requirements.txt
C:\polymarket\.venv\Scripts\python.exe -m pytest          # run from this folder

# Collect data for the most active market (read-only, no keys needed)
C:\polymarket\.venv\Scripts\python.exe -m polyagents
```

No API keys are required for the data layer (Gamma, prices-history, /trades and
the CLOB order book are all public read endpoints). Set `TAVILY_API_KEY` to
enable the news collector; without it the news report degrades gracefully.

## Design notes

- **Blackboard over message-passing.** Like TradingAgents, every node returns a
  dict that LangGraph merges into the single `MarketState`. Collectors are
  deterministic (no LLM) — they belong to the data layer, the LLM analyst
  agents read these reports later.
- **Reports carry both text and numbers.** Each collector writes a
  human-readable `*_report` string *and* structured numeric data into
  `state["raw"]`, because Polymarket's downstream (detectors, ML, sizing) needs
  the numbers, not just prose.
