"""
Adapter 2: Polymarket WebSocket
Subscribes to the Polymarket CLOB WS price feed for active markets.
Emits Signal(source_id=polymarket_ws, category=price) per trade tick.
Publishes to redis channel: oracle:signal

Protocol:
  1. Fetch top N active markets via REST to build token_id → market info map.
  2. Connect to WS, subscribe to those token_ids.
  3. Stream price_change / last_trade_price events → Signal → Redis.
  4. On disconnect: reconnect with exponential back-off, refreshing token map.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import websockets
import websockets.exceptions

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

CLOB_REST_BASE = "https://clob.polymarket.com"
CLOB_WS_URL    = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_CURSOR_END    = "LTE="

# WS subscription batch size (Polymarket max per message is undocumented; 500 is safe)
_BATCH = 500


class PolymarketWSAdapter:
    """
    Shared adapter interface:
        start()  — connect, subscribe, stream (runs until stop())
        stop()   — signal the loop to exit
    """

    def __init__(
        self,
        redis_client: Any,
        max_markets: int = 200,
        reconnect_delay: float = 5.0,
    ) -> None:
        self._redis = redis_client
        self._max_markets = max_markets
        self._reconnect_delay = reconnect_delay
        self._running = False
        # token_id → {"condition_id": str, "outcome": str}
        self._token_map: dict[str, dict[str, str]] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info(
            "PolymarketWSAdapter started  max_markets=%d", self._max_markets
        )
        delay = self._reconnect_delay
        while self._running:
            try:
                await self._connect_and_stream()
                delay = self._reconnect_delay  # reset on clean exit
            except (
                websockets.exceptions.WebSocketException,
                ConnectionError,
                OSError,
            ) as exc:
                if not self._running:
                    break
                logger.warning(
                    "PolymarketWSAdapter: connection error (%s) — retry in %gs",
                    exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)  # cap at 60s
            except Exception:
                if not self._running:
                    break
                logger.exception("PolymarketWSAdapter: unexpected error — retry in %gs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def stop(self) -> None:
        self._running = False
        logger.info("PolymarketWSAdapter stopped")

    # ── Core connect + stream ─────────────────────────────────────────────────

    async def _connect_and_stream(self) -> None:
        await self._refresh_token_map()

        token_ids = list(self._token_map.keys())
        if not token_ids:
            logger.warning("PolymarketWSAdapter: no token_ids — nothing to subscribe to")
            return

        logger.info(
            "PolymarketWSAdapter: connecting to WS  tokens=%d", len(token_ids)
        )

        async with websockets.connect(CLOB_WS_URL, ping_interval=20, ping_timeout=30) as ws:
            # Subscribe in batches
            for i in range(0, len(token_ids), _BATCH):
                batch = token_ids[i : i + _BATCH]
                await ws.send(
                    json.dumps({
                        "type": "subscribe",
                        "channel": "market",
                        "assets_ids": batch,
                    })
                )
            logger.info("PolymarketWSAdapter: subscribed to %d tokens", len(token_ids))

            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    await self._handle_message(raw_msg)
                except Exception:
                    logger.exception("PolymarketWSAdapter: error handling message")

    # ── Message handling ──────────────────────────────────────────────────────

    async def _handle_message(self, raw: str | bytes) -> None:
        data = json.loads(raw)
        # Messages arrive as a JSON array of event objects
        events = data if isinstance(data, list) else [data]
        for event in events:
            signal = self._normalize(event)
            if signal is not None:
                await self._publish(signal)

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize(self, event: dict[str, Any]) -> Signal | None:
        """
        Convert a raw WS event to a Signal.

        Handles two event types:
          price_change  — emits one signal per change entry
          last_trade_price — emits one signal for the trade

        raw_payload keys (per signal.md):
            market_id, outcome, price (float), side (buy/sell), size (float)
        """
        event_type = event.get("event_type", "")
        asset_id   = event.get("asset_id", "")
        token_info = self._token_map.get(asset_id, {})
        condition_id = token_info.get("condition_id", asset_id)
        outcome      = token_info.get("outcome", "")

        if event_type == "last_trade_price":
            return Signal(
                source_id=SourceId.POLYMARKET_WS,
                timestamp=datetime.now(timezone.utc),
                category=SignalCategory.PRICE,
                raw_payload={
                    "market_id": condition_id,
                    "outcome":   outcome,
                    "price":     _to_float(event.get("price")),
                    "side":      event.get("side", ""),
                    "size":      _to_float(event.get("size")),
                },
                market_ids=[condition_id] if condition_id else None,
            )

        if event_type == "price_change":
            changes: list[dict] = event.get("changes", [])
            if not changes:
                return None
            # Emit one signal for the first (most recent) change
            change = changes[0]
            return Signal(
                source_id=SourceId.POLYMARKET_WS,
                timestamp=datetime.now(timezone.utc),
                category=SignalCategory.PRICE,
                raw_payload={
                    "market_id": condition_id,
                    "outcome":   outcome,
                    "price":     _to_float(change.get("price")),
                    "side":      change.get("side", ""),
                    "size":      _to_float(change.get("size")),
                },
                market_ids=[condition_id] if condition_id else None,
            )

        return None  # book, tick_size_change, etc. — not price signals

    # ── Token map refresh ─────────────────────────────────────────────────────

    async def _refresh_token_map(self) -> None:
        """
        Fetch up to max_markets active markets from REST and build
        token_id → {condition_id, outcome} map.
        """
        self._token_map = {}
        markets_seen = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            cursor: str | None = None
            while markets_seen < self._max_markets:
                params: dict[str, Any] = {}
                if cursor:
                    params["next_cursor"] = cursor

                resp = await client.get(f"{CLOB_REST_BASE}/markets", params=params)
                resp.raise_for_status()
                body = resp.json()

                for market in body.get("data", []):
                    if not (market.get("active") and not market.get("closed")):
                        continue
                    cid = market.get("condition_id", "")
                    for token in market.get("tokens", []):
                        tid = token.get("token_id")
                        if tid:
                            self._token_map[tid] = {
                                "condition_id": cid,
                                "outcome":      token.get("outcome", ""),
                            }
                    markets_seen += 1
                    if markets_seen >= self._max_markets:
                        break

                next_cursor = body.get("next_cursor", _CURSOR_END)
                if next_cursor == _CURSOR_END or not body.get("data"):
                    break
                cursor = next_cursor

        logger.info(
            "PolymarketWSAdapter token map: %d tokens from %d markets",
            len(self._token_map), markets_seen,
        )

    # ── Publish ───────────────────────────────────────────────────────────────

    async def _publish(self, signal: Signal) -> None:
        """Publish a normalized Signal to the canonical Redis channel."""
        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
