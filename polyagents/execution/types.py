"""Execution-layer domain types.

Small, explicit value objects — Order / Fill / Position / ExecutionResult —
shared across the paper and live execution clients. Mirrors NautilusTrader's
clean separation of order intent from fill outcome, scaled down to Polymarket's
single-venue binary world.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

Side = Literal["buy", "sell"]
ExecStatus = Literal["filled", "blocked", "skipped", "rejected", "error"]


@dataclass
class Order:
    token_id: str
    side: Side
    size_usdc: float          # notional to deploy (buy); ignored for a full-exit sell
    ref_price: float          # reference price: best ask (buy) / best bid (sell)
    market: str = ""          # human label for logs
    book: Any = None          # live OrderBook (bids/asks levels) for realistic fills


@dataclass
class Fill:
    token_id: str
    side: Side
    price: float              # actual fill price (after modeled slippage)
    shares: float
    notional: float
    ts: datetime


@dataclass
class ExecutionResult:
    status: ExecStatus
    order: Optional[Order] = None
    fill: Optional[Fill] = None
    realized_pnl: float = 0.0
    reason: str = ""


@dataclass
class Position:
    token_id: str
    shares: float
    avg_price: float
    market: str = ""

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_price

    def mark_value(self, price: float) -> float:
        return self.shares * price

    def unrealized(self, price: float) -> float:
        return (price - self.avg_price) * self.shares
