"""Task 9 — SOE floor estimate handler.

Subscribes to oracle:re_floor_request. On each request, calls Claude Sonnet
with the asset's price history and returns a probabilistic floor estimate.
Publishes response to oracle:re_floor_response:{request_id}.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from oracle_shared.providers import LLMProvider, get_llm

from reasoning_engine.config import (
    FLOOR_REQUEST_CHANNEL,
    FLOOR_RESPONSE_PREFIX,
)

logger = logging.getLogger(__name__)

FLOOR_SYSTEM_PROMPT = """You are a quantitative analyst. Given a Solana asset's recent price \
history, estimate a probabilistic 24-72 hour price floor.

Return ONLY valid JSON:
{"floor_estimate_usd": <float>, "confidence": <float 0-1>, "reasoning": "<1-2 sentences>", "horizon_hours": <int>}"""


class FloorEstimator:
    """Handle SOE floor estimate requests via Redis pub/sub."""

    def __init__(self, redis_client: Any, llm: LLMProvider | None = None) -> None:
        self._redis = redis_client
        self._llm = llm or get_llm()
        self._running = False

    async def start(self) -> None:
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(FLOOR_REQUEST_CHANNEL)
        logger.info("FloorEstimator: subscribed to %s", FLOOR_REQUEST_CHANNEL)

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
                    request = json.loads(msg["data"])
                    await self._handle(request)
                except Exception:
                    logger.warning(
                        "FloorEstimator: failed to handle request", exc_info=True
                    )
        finally:
            await pubsub.unsubscribe(FLOOR_REQUEST_CHANNEL)
            await pubsub.aclose()

    async def _handle(self, request: dict[str, Any]) -> None:
        request_id = request.get("request_id", "")
        asset = request.get("asset", "unknown")
        price_history = request.get("price_history", [])

        # Build prompt
        prices_str = ", ".join(f"${p:.4f}" for p in price_history[-20:])
        user_prompt = (
            f"Asset: {asset}\n"
            f"Recent prices (newest last): [{prices_str}]\n"
            f"Estimate the 24-72h price floor."
        )

        try:
            result = await self._llm.generate_json(user_prompt, system=FLOOR_SYSTEM_PROMPT, max_tokens=256)
        except Exception:
            logger.warning("FloorEstimator: LLM call failed for %s", asset, exc_info=True)
            result = {
                "floor_estimate_usd": 0.0,
                "confidence": 0.0,
                "reasoning": "Estimation failed",
                "horizon_hours": 48,
            }

        result["request_id"] = request_id
        result["asset"] = asset

        # Publish response
        response_channel = f"{FLOOR_RESPONSE_PREFIX}:{request_id}"
        await self._redis.publish(response_channel, json.dumps(result))
        logger.info(
            "FloorEstimator: responded to %s  asset=%s  floor=$%.4f",
            request_id[:8], asset, result.get("floor_estimate_usd", 0),
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("FloorEstimator: stopped")
