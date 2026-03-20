"""Step 2 — Signal subscriber and text extraction.

Subscribes to oracle:signal. Extracts raw_text from each signal based on
source_id. Discards signals with empty or short text (< 10 chars).
Forwards qualifying signals + raw_text to the pipeline callback.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

# Callback type: (signal, raw_text) -> None
PipelineCallback = Callable[[Signal, str], Awaitable[None]]

# Minimum text length to process
MIN_TEXT_LENGTH = 10

# Text extraction rules by source_id
_TEXT_EXTRACTORS: dict[str, Callable[[dict[str, Any]], str]] = {
    SourceId.NEWSAPI.value: lambda rp: (
        (rp.get("title", "") or "") + ". " + (rp.get("description", "") or "")
    ).strip(". "),
    SourceId.WIKIPEDIA.value: lambda rp: (
        (rp.get("page_title", "") or "") + ": " + (rp.get("summary", "") or "")
    ).strip(": "),
    SourceId.REDDIT.value: lambda rp: rp.get("title", ""),
    SourceId.POLYMARKET_REST.value: lambda rp: rp.get("question", ""),
    SourceId.AI_OPINION.value: lambda rp: rp.get("response_text", ""),
}

# Source IDs to skip (no text content)
_SKIP_SOURCES = {
    SourceId.POLYGON_CLOB.value,
    SourceId.POLYMARKET_WS.value,
    SourceId.BIRDEYE.value,
}


class SignalSubscriber:
    """Subscribe to Signal.CHANNEL and forward text-bearing signals."""

    def __init__(
        self,
        redis_client: Any,
        on_signal: PipelineCallback,
    ) -> None:
        self._redis = redis_client
        self._on_signal = on_signal
        self._running = False

    async def start(self) -> None:
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(Signal.CHANNEL)
        logger.info("OSFE SignalSubscriber: subscribed to %s", Signal.CHANNEL)

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
                    signal = Signal.model_validate_json(msg["data"])
                except Exception:
                    logger.warning("OSFE SignalSubscriber: failed to parse signal", exc_info=True)
                    continue

                # Skip sources with no text content
                if signal.source_id.value in _SKIP_SOURCES:
                    continue

                # Extract text
                extractor = _TEXT_EXTRACTORS.get(signal.source_id.value)
                if extractor is None:
                    continue

                raw_text = extractor(signal.raw_payload).strip()
                if len(raw_text) < MIN_TEXT_LENGTH:
                    continue

                await self._on_signal(signal, raw_text)
        finally:
            await pubsub.unsubscribe(Signal.CHANNEL)
            await pubsub.aclose()

    async def stop(self) -> None:
        self._running = False
        logger.info("OSFE SignalSubscriber: stopped")
