"""Step 8 — PostMortem weight updater.

Subscribes to oracle:post_mortem. For each PostMortem received, applies
source_weight_updates deltas to the credibility weights in Redis.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from oracle_shared.contracts.post_mortem import PostMortem

from osint_fusion.credibility import CredibilityWeighter

logger = logging.getLogger(__name__)


class PostMortemSubscriber:
    """Subscribe to PostMortem channel and update credibility weights."""

    def __init__(
        self,
        redis_client: Any,
        credibility: CredibilityWeighter,
    ) -> None:
        self._redis = redis_client
        self._credibility = credibility
        self._running = False

    async def start(self) -> None:
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(PostMortem.CHANNEL)
        logger.info(
            "PostMortemSubscriber: subscribed to %s", PostMortem.CHANNEL
        )

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
                    pm = PostMortem.model_validate_json(msg["data"])
                    await self._process(pm)
                except Exception:
                    logger.warning(
                        "PostMortemSubscriber: failed to process message",
                        exc_info=True,
                    )
        finally:
            await pubsub.unsubscribe(PostMortem.CHANNEL)
            await pubsub.aclose()

    async def _process(self, pm: PostMortem) -> None:
        """Apply source_weight_updates from a PostMortem."""
        if not pm.source_weight_updates:
            return

        for source_id, delta in pm.source_weight_updates.items():
            await self._credibility.apply_delta(source_id, delta)

        logger.info(
            "PostMortemSubscriber: applied %d weight updates from PM %s",
            len(pm.source_weight_updates),
            pm.postmortem_id,
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("PostMortemSubscriber: stopped")
