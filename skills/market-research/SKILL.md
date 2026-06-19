---
name: market-research
description: Research a Polymarket question or theme WITHOUT trading — pull the data, find similar past markets (RAG), and explain the state of a market. Use when the user wants analysis or understanding, not execution.
---

# Market research

Help the user understand a market or theme. **Do not place trades** in this mode.

## Workflow
- `scan_markets` to find relevant markets, `market_snapshot` for the evidence
  (price/volume, order-book microstructure, trade-flow imbalance, factors).
- `find_similar_markets` for historical context — semantically similar past
  markets and how they resolved.
- Summarise: what the microstructure and flow say, what would move the price, and
  how confident the read is. Be explicit about uncertainty.

## Discipline
- Explain, don't execute. If the user wants to act on the analysis, tell them to
  enable the **polymarket-trading** skill.
- Never invent numbers — only state what the tools return.
