"""
Adapter 6: AI Opinion Poller
Queries the Reasoning Engine on a scheduled interval via Redis request/reply
for market outlook. Emits Signal(source_id=ai_opinion, category=ai_generated)
per response.
Publishes to redis channel: oracle:signal

The RE is expected to subscribe to oracle:re:request and publish replies
to oracle:re:response:{request_id}. If RE is not running, the adapter
logs a warning and retries on the next scheduled tick.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

RE_REQUEST_CHANNEL = "oracle:re:request"
RE_RESPONSE_PREFIX = "oracle:re:response:"
RE_RESPONSE_TIMEOUT = 60  # seconds to wait for RE reply


class AIOpinionAdapter:
    """
    Shared adapter interface:
        start()  — begin scheduled polling loop (runs until stop())
        stop()   — signal the loop to exit
    """

    def __init__(
        self,
        redis_client: Any,
        poll_interval: int | None = None,
    ) -> None:
        self._redis = redis_client
        self._poll_interval = poll_interval or int(
            os.getenv("AI_OPINION_POLL_INTERVAL", "1800")
        )
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info(
            "AIOpinionAdapter started  poll_interval=%ds", self._poll_interval
        )
        while self._running:
            try:
                await self._poll()
            except Exception:
                logger.exception("AIOpinionAdapter: unhandled error in poll()")
            if self._running:
                await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("AIOpinionAdapter stopped")

    # ── Core poll ─────────────────────────────────────────────────────────────

    async def _poll(self) -> None:
        # Fetch active market IDs from Redis state
        market_ids = await self._get_active_market_ids()
        if not market_ids:
            logger.info("AIOpinionAdapter: no active markets — skipping")
            return

        published = 0
        for market_id in market_ids:
            signal = await self._query_re(market_id)
            if signal is not None:
                await self._publish(signal)
                published += 1

        logger.info("AIOpinionAdapter: published %d signals", published)

    async def _get_active_market_ids(self) -> list[str]:
        """Read active market IDs from oracle:state:markets Redis hash."""
        try:
            keys = await self._redis.hkeys("oracle:state:markets")
            return list(keys)[:20]  # cap to avoid flooding RE
        except Exception:
            logger.warning("AIOpinionAdapter: could not read oracle:state:markets")
            return []

    # ── RE request/reply ──────────────────────────────────────────────────────

    async def _query_re(self, market_id: str) -> Signal | None:
        """
        Send an opinion request to RE and await a response.
        Returns None if RE does not respond within timeout.
        """
        request_id = str(uuid.uuid4())
        response_channel = f"{RE_RESPONSE_PREFIX}{request_id}"

        # Subscribe before publishing to avoid race
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(response_channel)

        try:
            # Publish request
            request = json.dumps({
                "request_id": request_id,
                "type": "opinion",
                "market_id": market_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await self._redis.publish(RE_REQUEST_CHANNEL, request)

            # Wait for response
            deadline = asyncio.get_event_loop().time() + RE_RESPONSE_TIMEOUT
            while asyncio.get_event_loop().time() < deadline:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if msg and msg["type"] == "message":
                    return self._normalize(
                        json.loads(msg["data"]), market_id, request_id
                    )

            logger.warning(
                "AIOpinionAdapter: RE did not respond for market %s within %ds",
                market_id,
                RE_RESPONSE_TIMEOUT,
            )
            return None
        finally:
            await pubsub.unsubscribe(response_channel)
            await pubsub.aclose()

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize(
        self,
        response: dict[str, Any],
        market_id: str,
        request_id: str,
    ) -> Signal:
        """
        raw_payload keys (per signal.md):
            model, prompt_used, response_text, market_ids_queried
        """
        return Signal(
            source_id=SourceId.AI_OPINION,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.AI,
            raw_payload={
                "model": response.get("model", "unknown"),
                "prompt_used": response.get("prompt", ""),
                "response_text": response.get("response", ""),
                "market_ids_queried": [market_id],
            },
            confidence=_to_float(response.get("confidence")),
            market_ids=[market_id],
        )

    # ── Publish ───────────────────────────────────────────────────────────────

    async def _publish(self, signal: Signal) -> None:
        """Publish a normalized Signal to the canonical Redis channel."""
        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
