"""
Adapter 1: Polymarket REST
Polls GET /markets every 60 s (configurable).
Emits Signal(source_id=polymarket_rest, category=price) per active market.
Publishes to redis channel: oracle:signal
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"
_CURSOR_END = "LTE="  # Polymarket sentinel: no more pages


class PolymarketRESTAdapter:
    """
    Shared adapter interface:
        start()  — begin polling loop (runs until stop() is called)
        stop()   — signal the loop to exit after the current iteration
    """

    def __init__(self, redis_client: Any, poll_interval: int = 60) -> None:
        self._redis = redis_client
        self._poll_interval = poll_interval
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info(
            "PolymarketRESTAdapter started  poll_interval=%ds", self._poll_interval
        )
        while self._running:
            try:
                await self._poll()
            except Exception:
                logger.exception("PolymarketRESTAdapter: unhandled error in poll()")
            if self._running:
                await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("PolymarketRESTAdapter stopped")

    # ── Core poll ─────────────────────────────────────────────────────────────

    async def _poll(self) -> None:
        published = 0
        async for market in self._fetch_active_markets():
            signal = self._normalize(market)
            await self._publish(signal)
            published += 1
        logger.info("PolymarketRESTAdapter: published %d signals", published)

    # ── Fetch ─────────────────────────────────────────────────────────────────

    async def _fetch_active_markets(self) -> AsyncIterator[dict[str, Any]]:
        """
        Cursor-paginated walk of GET /markets.
        The API ignores active/closed query params, so we filter client-side:
        only yield markets where active=True and closed=False.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            cursor: str | None = None  # omit on first page
            while True:
                params: dict[str, Any] = {}
                if cursor:
                    params["next_cursor"] = cursor

                resp = await client.get(f"{CLOB_BASE}/markets", params=params)
                resp.raise_for_status()
                body = resp.json()

                markets: list[dict[str, Any]] = body.get("data", [])
                for market in markets:
                    if market.get("active") and not market.get("closed"):
                        yield market

                next_cursor: str = body.get("next_cursor", _CURSOR_END)
                if next_cursor == _CURSOR_END or not markets:
                    break
                cursor = next_cursor

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize(self, market: dict[str, Any]) -> Signal:
        """
        Raw Polymarket market dict → canonical Signal.

        raw_payload keys (per signal.md contract):
            market_id       — condition_id
            question        — market question text
            outcome_prices  — {outcome_label: price_float, ...}  e.g. {"Yes": 0.72, "No": 0.28}
            volume_usd      — float
            liquidity_usd   — float
            end_date        — ISO-8601 string from end_date_iso
        """
        tokens: list[dict[str, Any]] = market.get("tokens", [])
        outcome_prices: dict[str, float] = {
            tok.get("outcome", f"token_{i}"): float(tok.get("price", 0.0) or 0.0)
            for i, tok in enumerate(tokens)
        }

        condition_id: str = market.get("condition_id", "")

        return Signal(
            source_id=SourceId.POLYMARKET_REST,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.PRICE,
            raw_payload={
                "market_id": condition_id,
                "question": market.get("question", ""),
                "outcome_prices": outcome_prices,
                "volume_usd": _to_float(market.get("volume")),
                "liquidity_usd": _to_float(market.get("liquidity")),
                "end_date": market.get("end_date_iso", ""),
            },
            market_ids=[condition_id] if condition_id else None,
        )

    # ── Publish ───────────────────────────────────────────────────────────────

    async def _publish(self, signal: Signal) -> None:
        """Publish a normalized Signal to the canonical Redis channel."""
        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(value: Any) -> float:
    """Safely coerce a possibly-string numeric field to float."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
