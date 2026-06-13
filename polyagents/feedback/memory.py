"""Persistent decision log — the memory behind the feedback loop.

Mirrors TradingAgents' always-on decision log: every trade is appended here, and
once it resolves we record the realised return + a one-paragraph lesson. Recent
lessons are injected back into the signal agent so each analysis carries forward
what worked and what didn't.

Stored as JSONL at ``~/.polyagents/memory/trades.jsonl`` (override via config).
Append for new records; rewrite for in-place outcome updates (paper scale).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from polyagents.dataflows.utils import utcnow


def make_trade_record(state: dict) -> dict[str, Any]:
    """Build a pending record from a finished trading-graph state."""
    decision = state["trade_decision"]
    signal = state.get("signal")
    exec_result = state.get("execution_result")
    return {
        "record_id": uuid.uuid4().hex[:12],
        "ts": utcnow().isoformat(),
        "status": "pending",
        "market_id": state.get("market_id", ""),
        "condition_id": state.get("condition_id", ""),
        "token_id": state.get("token_id", ""),
        "question": state.get("question", ""),
        "side": state.get("outcome", ""),
        "action": decision.action,
        "p_true": decision.p_true,                 # calibrated p used to size
        "raw_p_true": getattr(decision, "raw_p_true", None),
        "market_price": decision.market_price,
        "edge": decision.edge,
        "annualized_edge": getattr(decision, "annualized_edge", None),
        "days_to_expiry": getattr(decision, "days_to_expiry", None),
        "size_usdc": decision.size_usdc,
        "exec_status": getattr(exec_result, "status", None),
        "fill_price": getattr(getattr(exec_result, "fill", None), "price", None),
        "signal_rationale": getattr(signal, "rationale", ""),
        # filled in at settlement:
        "resolved_winner": None,
        "won": None,
        "realized_pnl": None,
        "realized_return": None,
        "lesson": None,
    }


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ----- reads -------------------------------------------------------------

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def pending(self) -> list[dict]:
        return [r for r in self.all() if r.get("status") == "pending"]

    def recent_lessons(self, question: str | None = None, limit: int = 3) -> list[str]:
        """Most-recent resolved lessons, same-market first then cross-market."""
        resolved = [r for r in self.all() if r.get("lesson")]
        same = [r for r in resolved if question and r.get("question") == question]
        others = [r for r in resolved if r not in same]
        picked = (same[::-1] + others[::-1])[:limit]
        return [r["lesson"] for r in picked]

    # ----- writes ------------------------------------------------------------

    def record(self, rec: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def update(self, record_id: str, **fields) -> None:
        rows = self.all()
        for r in rows:
            if r.get("record_id") == record_id:
                r.update(fields)
        self.path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
