"""Task 2 — Insight subscriber.

Subscribes to oracle:insight. For each Insight, enqueues a trigger analysis
for each associated market. Deduplicates using a Redis set.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from oracle_shared.contracts.insight import Insight

from reasoning_engine.config import RE_QUEUE_KEY

logger = logging.getLogger(__name__)

# Callback: async def on_market(market_id: str) -> None
MarketCallback = Callable[[str], Awaitable[None]]


class InsightSubscriber:
    """Subscribe to Insight.CHANNEL and enqueue market analysis triggers."""

    def __init__(
        self,
        redis_client: Any,
        on_market: MarketCallback,
    ) -> None:
        self._redis = redis_client
        self._on_market = on_market
        self._running = False

    async def start(self) -> None:
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(Insight.CHANNEL)
        logger.info("InsightSubscriber: subscribed to %s", Insight.CHANNEL)

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

                try:
                    insight = Insight.model_validate_json(msg["data"])
                except Exception:
                    logger.warning("InsightSubscriber: failed to parse", exc_info=True)
                    continue

                for market_id in insight.associated_market_ids:
                    # Deduplicate: only trigger if not already queued
                    added = await self._redis.sadd(RE_QUEUE_KEY, market_id)
                    if added:
                        logger.debug(
                            "InsightSubscriber: enqueued market %s", market_id
                        )
                        await self._on_market(market_id)
                    # Remove from queue after triggering (fire-once per signal batch)
                    await self._redis.srem(RE_QUEUE_KEY, market_id)
        finally:
            await pubsub.unsubscribe(Insight.CHANNEL)
            await pubsub.aclose()

    async def stop(self) -> None:
        self._running = False
        logger.info("InsightSubscriber: stopped")
