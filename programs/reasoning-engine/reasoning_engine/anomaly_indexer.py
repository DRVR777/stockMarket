"""Task 10 — AnomalyEvent indexer.

Subscribes to oracle:anomaly_event. On each event, LPUSHes the event JSON
to oracle:state:anomaly_index:{market_id} and trims to last 10.
Also persists to Postgres.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from oracle_shared.contracts.anomaly_event import AnomalyEvent
from oracle_shared.db import get_session
from oracle_shared.db.repository import AnomalyEventRepo

from reasoning_engine.config import ANOMALY_INDEX_PREFIX

logger = logging.getLogger(__name__)


class AnomalyIndexer:
    """Index AnomalyEvents per market in Redis and persist to Postgres."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._running = False

    async def start(self) -> None:
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(AnomalyEvent.CHANNEL)
        logger.info("AnomalyIndexer: subscribed to %s", AnomalyEvent.CHANNEL)

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
                    event = AnomalyEvent.model_validate_json(msg["data"])
                    await self._index(event, msg["data"])
                except Exception:
                    logger.warning("AnomalyIndexer: failed to process", exc_info=True)
        finally:
            await pubsub.unsubscribe(AnomalyEvent.CHANNEL)
            await pubsub.aclose()

    async def _index(self, event: AnomalyEvent, raw_json: str) -> None:
        """LPUSH to Redis list and persist to Postgres."""
        key = f"{ANOMALY_INDEX_PREFIX}:{event.market_id}"
        await self._redis.lpush(key, raw_json)
        await self._redis.ltrim(key, 0, 9)  # keep last 10

        try:
            async with get_session() as session:
                await AnomalyEventRepo.save(session, event)
        except Exception:
            logger.warning("AnomalyIndexer: Postgres save failed", exc_info=True)

        logger.debug("AnomalyIndexer: indexed event %s for market %s",
                      event.event_id, event.market_id)

    async def stop(self) -> None:
        self._running = False
        logger.info("AnomalyIndexer: stopped")
