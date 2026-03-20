"""Task 2 — Redis subscriber bridge.

Subscribes to all oracle:* channels. On each message, broadcasts
to all connected WebSocket clients as typed JSON events.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

from operator_dashboard.config import ALL_CHANNELS

logger = logging.getLogger(__name__)

# Channel name -> event type for the frontend
_CHANNEL_TYPE_MAP = {
    "oracle:signal": "signal",
    "oracle:anomaly_event": "anomaly_event",
    "oracle:insight": "insight",
    "oracle:market_state": "market_state",
    "oracle:trade_thesis": "trade_thesis",
    "oracle:trade_execution": "trade_execution",
    "oracle:post_mortem": "post_mortem",
    "oracle:operator_alert": "operator_alert",
}


class RedisBridge:
    """Bridge Redis pub/sub events to WebSocket clients."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._clients: set[WebSocket] = set()
        self._running = False

    def add_client(self, ws: WebSocket) -> None:
        self._clients.add(ws)
        logger.info("RedisBridge: client connected  total=%d", len(self._clients))

    def remove_client(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info("RedisBridge: client disconnected  total=%d", len(self._clients))

    async def start(self) -> None:
        """Subscribe to all channels and broadcast to WS clients."""
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(*ALL_CHANNELS)
        logger.info("RedisBridge: subscribed to %d channels", len(ALL_CHANNELS))

        try:
            while self._running:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if msg is None:
                    await asyncio.sleep(0.05)
                    continue
                if msg["type"] != "message":
                    continue

                channel = msg["channel"]
                event_type = _CHANNEL_TYPE_MAP.get(channel, "unknown")

                try:
                    data = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    data = {"raw": msg["data"]}

                envelope = json.dumps({
                    "type": event_type,
                    "channel": channel,
                    "data": data,
                })

                await self._broadcast(envelope)
        finally:
            await pubsub.unsubscribe(*ALL_CHANNELS)
            await pubsub.aclose()

    async def _broadcast(self, message: str) -> None:
        """Send a message to all connected WebSocket clients."""
        disconnected: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self._clients.discard(ws)

    async def stop(self) -> None:
        self._running = False
        logger.info("RedisBridge: stopped")
