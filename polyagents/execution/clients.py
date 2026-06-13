"""Execution clients — the port and its adapters (NautilusTrader-style).

``ExecutionClient`` is the port; strategies depend only on it. Two adapters:

  * ``PaperExecutionClient`` — simulates fills against the touch price plus
    modeled slippage and books them into the :class:`Portfolio`. The default.
  * ``LiveCLOBExecutionClient`` — posts real GTC limit orders via the official
    py-clob-client (needs ``POLYMARKET_PRIVATE_KEY``). Gated; never the default.

Swapping paper ↔ live changes nothing upstream — same port, different adapter.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from .portfolio import Portfolio
from .types import ExecutionResult, Fill, Order


@runtime_checkable
class ExecutionClient(Protocol):
    def submit(self, order: Order, portfolio: Portfolio) -> ExecutionResult:
        ...


class PaperExecutionClient:
    """Simulated venue: fill at touch ± slippage, book into the portfolio."""

    def __init__(self, slippage_bps: float = 50.0) -> None:
        self.slippage_bps = slippage_bps

    def submit(self, order: Order, portfolio: Portfolio) -> ExecutionResult:
        """Fill by WALKING the live order book (realistic slippage + impact). A
        large order eats through levels, so the average fill is worse than the
        touch — modelling this matters: filling at mid/touch systematically
        overstates paper P&L, which then poisons the L4 feedback loop. Falls back
        to a fixed-slippage touch fill only when no book is attached."""
        now = datetime.now(timezone.utc)

        if order.side == "buy":
            if order.size_usdc <= 0:
                return ExecutionResult("skipped", order, reason="zero size")
            shares, notional, avg = self._buy_fill(order)
            if shares <= 0 or avg <= 0:
                return ExecutionResult("rejected", order, reason="no ask liquidity")
            fill = Fill(order.token_id, "buy", avg, shares, notional, now)
            portfolio.apply_buy(fill)
            slip = (avg / (order.ref_price or avg) - 1.0)
            partial = "" if notional >= order.size_usdc - 0.01 else f" (partial, book thin)"
            return ExecutionResult("filled", order, fill, 0.0,
                                   f"bought {shares:,.0f} @ avg {avg:.3f} (slip {slip:+.1%}){partial}")

        # sell = full exit of the held position
        if order.token_id not in portfolio.positions:
            return ExecutionResult("skipped", order, reason="no position to sell")
        pos = portfolio.positions[order.token_id]
        shares = pos.shares
        avg = self._sell_fill(order, shares)
        if avg <= 0:
            return ExecutionResult("rejected", order, reason="no bid liquidity")
        pnl = portfolio.apply_sell_close(order.token_id, avg, now)
        fill = Fill(order.token_id, "sell", avg, shares, avg * shares, now)
        slip = (1.0 - avg / (order.ref_price or avg))
        return ExecutionResult("filled", order, fill, pnl,
                               f"sold {shares:,.0f} @ avg {avg:.3f} (slip {slip:+.1%}), P&L {pnl:+,.2f}")

    # ----- fill models -------------------------------------------------------

    def _buy_fill(self, order: Order) -> tuple[float, float, float]:
        """Return (shares, notional_spent, avg_price) for a buy of size_usdc."""
        asks = list(getattr(order.book, "asks", None) or [])
        if not asks:                                   # no book → fixed-slippage touch fill
            price = order.ref_price * (1 + self.slippage_bps / 10_000.0)
            if price <= 0:
                return 0.0, 0.0, 0.0
            return order.size_usdc / price, order.size_usdc, price
        spent = shares = 0.0
        for lvl in asks:                               # best (lowest) ask first
            cap = lvl.price * lvl.size
            if spent + cap >= order.size_usdc:
                take = order.size_usdc - spent
                shares += take / lvl.price
                spent = order.size_usdc
                break
            spent += cap
            shares += lvl.size
        return shares, spent, (spent / shares if shares else 0.0)

    def _sell_fill(self, order: Order, shares_to_sell: float) -> float:
        """Average price selling ``shares_to_sell`` by walking the bids."""
        bids = list(getattr(order.book, "bids", None) or [])
        if not bids:
            return order.ref_price * (1 - self.slippage_bps / 10_000.0)
        proceeds = sold = 0.0
        worst = bids[-1].price
        for lvl in bids:                               # best (highest) bid first
            if sold + lvl.size >= shares_to_sell:
                proceeds += (shares_to_sell - sold) * lvl.price
                sold = shares_to_sell
                break
            proceeds += lvl.size * lvl.price
            sold += lvl.size
            worst = lvl.price
        if sold < shares_to_sell:                      # book too thin → remainder at worst level
            proceeds += (shares_to_sell - sold) * worst
        return proceeds / shares_to_sell if shares_to_sell else 0.0


class LiveCLOBExecutionClient:
    """Real GTC limit orders via the official SDK. Requires private-key creds."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._clob = None
        self._init_error = ""
        self._init()

    def _init(self) -> None:
        import os

        key = os.getenv("POLYMARKET_PRIVATE_KEY")
        if not key:
            self._init_error = "POLYMARKET_PRIVATE_KEY not set"
            return
        try:
            from py_clob_client.client import ClobClient

            client = ClobClient(
                host=self._config["clob_base"],
                chain_id=self._config.get("polymarket_chain_id", 137),
                key=key,
            )
            client.set_api_creds(client.create_or_derive_api_creds())
            self._clob = client
        except Exception as exc:  # pragma: no cover - needs live creds
            self._init_error = f"CLOB init failed: {exc}"

    def submit(self, order: Order, portfolio: Portfolio) -> ExecutionResult:  # pragma: no cover - live only
        if self._clob is None:
            return ExecutionResult("error", order, reason=f"live disabled: {self._init_error}")
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            price = order.ref_price
            shares = order.size_usdc / price if order.side == "buy" else portfolio.positions[order.token_id].shares
            signed = self._clob.create_order(
                OrderArgs(token_id=order.token_id, price=price, size=shares, side=order.side.upper())
            )
            resp = self._clob.post_order(signed, OrderType.GTC)
            return ExecutionResult("filled", order, reason=f"posted GTC: {resp}")
        except Exception as exc:
            return ExecutionResult("error", order, reason=f"post failed: {exc}")
