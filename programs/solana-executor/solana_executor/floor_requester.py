"""Task 4 — RE floor estimate requester.

Every 6 hours per asset, requests an AI floor estimate from the
Reasoning Engine via Redis request/reply.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from solana_executor.config import (
    FLOOR_REQUEST_CHANNEL,
    FLOOR_RESPONSE_PREFIX,
)
from solana_executor.statistical_model import AssetModel, ModelStore

logger = logging.getLogger(__name__)

FLOOR_RESPONSE_TIMEOUT = 90  # seconds


class FloorRequester:
    """Request AI floor estimates from RE for each asset."""

    def __init__(
        self,
        redis_client: Any,
        model_store: ModelStore,
    ) -> None:
        self._redis = redis_client
        self._store = model_store

    async def request_floor(self, model: AssetModel) -> float | None:
        """Request a floor estimate from RE and wait for response.

        Returns the floor estimate USD value, or None on timeout.
        """
        request_id = str(uuid.uuid4())
        response_channel = f"{FLOOR_RESPONSE_PREFIX}:{request_id}"

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(response_channel)

        try:
            request = json.dumps({
                "request_id": request_id,
                "asset": model.symbol,
                "token_address": model.token_address,
                "chain": model.chain,
                "price_history": model.prices_30d[-20:],
                "current_price": model.current_price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await self._redis.publish(FLOOR_REQUEST_CHANNEL, request)
            logger.info(
                "FloorRequester: requested floor for %s (id=%s)",
                model.symbol, request_id[:8],
            )

            deadline = asyncio.get_event_loop().time() + FLOOR_RESPONSE_TIMEOUT
            while asyncio.get_event_loop().time() < deadline:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if msg and msg["type"] == "message":
                    data = json.loads(msg["data"])
                    floor = float(data.get("floor_estimate_usd", 0))

                    # Update model
                    model.ai_floor_estimate = floor
                    model.ai_floor_estimated_at = datetime.now(timezone.utc).isoformat()
                    await self._store.save(model)

                    logger.info(
                        "FloorRequester: %s floor=$%.4f  confidence=%.2f",
                        model.symbol, floor, data.get("confidence", 0),
                    )
                    return floor

            logger.warning(
                "FloorRequester: timeout for %s (no RE response in %ds)",
                model.symbol, FLOOR_RESPONSE_TIMEOUT,
            )
            return None
        finally:
            await pubsub.unsubscribe(response_channel)
            await pubsub.aclose()
