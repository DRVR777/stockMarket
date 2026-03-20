"""
Adapter 5: Birdeye Price Adapter
Subscribes to Birdeye WebSocket for configured Solana token addresses.
Falls back to REST polling at /defi/price every 5s if WS disconnects.
Emits Signal(source_id=birdeye, category=price) per price update.
Publishes to redis channel: oracle:signal
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
import websockets
import websockets.exceptions

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

BIRDEYE_WS_URL = "wss://public-api.birdeye.so/socket"
BIRDEYE_REST_BASE = "https://public-api.birdeye.so"

# Well-known Solana tokens to monitor by default
DEFAULT_TOKENS: dict[str, str] = {
    "So11111111111111111111111111111111111111112": "SOL",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN": "JUP",
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "BONK",
}


class BirdeyeWSAdapter:
    """
    Shared adapter interface:
        start()  — connect, subscribe, stream (runs until stop())
        stop()   — signal the loop to exit
    """

    def __init__(
        self,
        redis_client: Any,
        api_key: str | None = None,
        tokens: dict[str, str] | None = None,
        reconnect_delay: float = 5.0,
        rest_fallback_interval: float = 5.0,
    ) -> None:
        self._redis = redis_client
        self._api_key = api_key or os.getenv("BIRDEYE_API_KEY", "")
        self._tokens = tokens or DEFAULT_TOKENS
        self._reconnect_delay = reconnect_delay
        self._rest_fallback_interval = rest_fallback_interval
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        if not self._api_key:
            logger.error("BirdeyeWSAdapter: BIRDEYE_API_KEY not set — skipping")
            return
        logger.info(
            "BirdeyeWSAdapter started  tokens=%d", len(self._tokens)
        )
        delay = self._reconnect_delay
        while self._running:
            try:
                await self._connect_and_stream()
                delay = self._reconnect_delay
            except (
                websockets.exceptions.WebSocketException,
                ConnectionError,
                OSError,
            ) as exc:
                if not self._running:
                    break
                logger.warning(
                    "BirdeyeWSAdapter: WS error (%s) — falling back to REST polling",
                    exc,
                )
                await self._rest_fallback_loop()
                delay = self._reconnect_delay
            except Exception:
                if not self._running:
                    break
                logger.exception(
                    "BirdeyeWSAdapter: unexpected error — retry in %gs", delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def stop(self) -> None:
        self._running = False
        logger.info("BirdeyeWSAdapter stopped")

    # ── WebSocket stream ──────────────────────────────────────────────────────

    async def _connect_and_stream(self) -> None:
        headers = {"X-API-KEY": self._api_key}
        async with websockets.connect(
            BIRDEYE_WS_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=30,
        ) as ws:
            # Subscribe to price updates for each token
            for address in self._tokens:
                await ws.send(json.dumps({
                    "type": "SUBSCRIBE_PRICE",
                    "data": {"address": address, "chain": "solana"},
                }))

            logger.info(
                "BirdeyeWSAdapter: subscribed to %d tokens via WS",
                len(self._tokens),
            )

            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(raw_msg)
                    signal = self._normalize_ws(data)
                    if signal is not None:
                        await self._publish(signal)
                except Exception:
                    logger.exception("BirdeyeWSAdapter: error handling WS message")

    def _normalize_ws(self, data: dict[str, Any]) -> Signal | None:
        """Normalize a Birdeye WS price event."""
        event_data = data.get("data", data)
        address = event_data.get("address", "")
        if not address:
            return None

        symbol = self._tokens.get(address, address[:8])
        return Signal(
            source_id=SourceId.BIRDEYE,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.PRICE,
            raw_payload={
                "token_address": address,
                "symbol": symbol,
                "price_usd": _to_float(event_data.get("price")),
                "price_change_24h_pct": _to_float(event_data.get("priceChange24h")),
                "volume_24h_usd": _to_float(event_data.get("volume24h")),
            },
        )

    # ── REST fallback ─────────────────────────────────────────────────────────

    async def _rest_fallback_loop(self) -> None:
        """Poll Birdeye REST API when WS is unavailable."""
        logger.info("BirdeyeWSAdapter: entering REST fallback mode")
        async with httpx.AsyncClient(timeout=30.0) as client:
            while self._running:
                try:
                    for address, symbol in self._tokens.items():
                        resp = await client.get(
                            f"{BIRDEYE_REST_BASE}/defi/price",
                            params={"address": address},
                            headers={
                                "X-API-KEY": self._api_key,
                                "x-chain": "solana",
                            },
                        )
                        resp.raise_for_status()
                        body = resp.json().get("data", {})
                        signal = Signal(
                            source_id=SourceId.BIRDEYE,
                            timestamp=datetime.now(timezone.utc),
                            category=SignalCategory.PRICE,
                            raw_payload={
                                "token_address": address,
                                "symbol": symbol,
                                "price_usd": _to_float(body.get("value")),
                                "price_change_24h_pct": _to_float(
                                    body.get("priceChange24h")
                                ),
                                "volume_24h_usd": _to_float(
                                    body.get("volume24h")
                                ),
                            },
                        )
                        await self._publish(signal)
                except Exception:
                    logger.exception("BirdeyeWSAdapter: REST fallback error")
                await asyncio.sleep(self._rest_fallback_interval)

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
