"""Step 1 — Signal subscriber.

Subscribes to the ``oracle:signal`` Redis pub/sub channel.
Filters for on-chain Polygon CLOB signals (``category == 'on_chain'`` AND
``source_id == 'polygon_clob'``). Passes matching signals to the detection
pipeline callback. All other categories are silently ignored.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

# Type alias for the pipeline callback
PipelineCallback = Callable[[Signal], Awaitable[None]]


class SignalSubscriber:
    """Subscribe to Signal.CHANNEL and forward qualifying signals to a callback."""

    def __init__(
        self,
        redis_client: Any,
        on_signal: PipelineCallback,
    ) -> None:
        self._redis = redis_client
        self._on_signal = on_signal
        self._running = False
        self._pubsub: Any = None

    async def start(self) -> None:
        """Begin listening for signals. Runs until ``stop()`` is called."""
        self._running = True
        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(Signal.CHANNEL)
        logger.info(
            "SignalSubscriber: subscribed to %s", Signal.CHANNEL,
        )

        try:
            while self._running:
                msg = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if msg is None:
                    await asyncio.sleep(0.05)
                    continue

                if msg["type"] != "message":
                    continue

                try:
                    data = json.loads(msg["data"])
                    signal = Signal.model_validate(data)
                except Exception:
                    logger.warning(
                        "SignalSubscriber: failed to parse signal message",
                        exc_info=True,
                    )
                    continue

                # Filter: only on_chain + polygon_clob
                if (
                    signal.category != SignalCategory.ON_CHAIN
                    or signal.source_id != SourceId.POLYGON_CLOB
                ):
                    continue

                logger.debug(
                    "SignalSubscriber: forwarding signal %s", signal.signal_id,
                )
                await self._on_signal(signal)
        finally:
            await self._pubsub.unsubscribe(Signal.CHANNEL)
            await self._pubsub.aclose()

    async def stop(self) -> None:
        """Signal the subscriber loop to exit."""
        self._running = False
        logger.info("SignalSubscriber: stopped")
