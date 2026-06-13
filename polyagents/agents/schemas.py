"""Structured outputs for the Layer 2 agents.

The LLM agents (signal, reflection) emit pydantic models via the provider's
structured-output mode — the same approach TradingAgents v0.2.4 adopted for its
Research Manager / Trader / PM. The decision is a plain dataclass because it is
produced by deterministic risk math, not an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["yes", "no", "none"]
Conviction = Literal["low", "medium", "high"]


class Signal(BaseModel):
    """Signal agent output: a probability read on the analysed outcome side."""

    direction: Direction = Field(
        description="Lean for the analysed side: 'yes' (will resolve YES), 'no', or 'none'."
    )
    p_true: float = Field(
        ge=0.0, le=1.0,
        description="Estimated TRUE probability the analysed side resolves YES (pays $1).",
    )
    conviction: Conviction = Field(description="Confidence in this estimate.")
    rationale: str = Field(description="2-4 sentences grounded in the provided factors/flow.")


class Reflection(BaseModel):
    """Reflection agent output: a pre-trade self-critique of the decision."""

    assessment: str = Field(description="Is the decision sound given the evidence? 2-4 sentences.")
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Concrete risks / shaky assumptions / out-of-distribution signs.",
    )
    confidence: Conviction = Field(description="Confidence that the decision is correct.")


class Lesson(BaseModel):
    """Reflection agent output after a trade resolves (the learning signal)."""

    summary: str = Field(description="One-paragraph lesson: what the signal got right/wrong and why.")
    what_to_change: str = Field(description="One concrete adjustment for next time on similar markets.")


@dataclass
class TradeDecision:
    """Deterministic decision agent output (risk + Kelly sizing).

    ``p_true`` is the CALIBRATED probability actually used for sizing;
    ``raw_p_true`` is the model's pre-calibration estimate.
    """

    action: Literal["buy", "sell", "hold"]
    p_true: float
    market_price: float
    edge: float
    kelly_fraction: float
    size_usdc: float
    reasons: list[str] = field(default_factory=list)
    annualized_edge: float = 0.0
    raw_p_true: float | None = None
    days_to_expiry: float | None = None
