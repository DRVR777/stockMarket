"""Task 6 — TradeExecution subscriber.

Subscribes to oracle:trade_execution.
- On open: updates market vault file.
- On closed: triggers post-mortem pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from oracle_shared.contracts.trade_execution import (
    ExecutionStatus,
    TradeExecution,
)

from knowledge_base.market_writer import MarketWriter

logger = logging.getLogger(__name__)

# Callback for triggering post-mortem: (market_id, thesis_id, execution) -> None
PostMortemCallback = Callable[[str, str | None, TradeExecution], Awaitable[None]]


class ExecutionSubscriber:
    """Subscribe to TradeExecution events and route to vault/post-mortem."""

    def __init__(
        self,
        redis_client: Any,
        market_writer: MarketWriter,
        on_close: PostMortemCallback,
    ) -> None:
        self._redis = redis_client
        self._market_writer = market_writer
        self._on_close = on_close
        self._running = False

    async def start(self) -> None:
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(TradeExecution.CHANNEL)
        logger.info("ExecutionSubscriber: subscribed to %s", TradeExecution.CHANNEL)

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
                    execution = TradeExecution.model_validate_json(msg["data"])
                    await self._process(execution)
                except Exception:
                    logger.warning("ExecutionSubscriber: failed to process", exc_info=True)
        finally:
            await pubsub.unsubscribe(TradeExecution.CHANNEL)
            await pubsub.aclose()

    async def _process(self, execution: TradeExecution) -> None:
        if execution.status == ExecutionStatus.OPEN:
            self._market_writer.add_execution(
                execution.market_id,
                execution.execution_id,
                execution.direction,
            )
        elif execution.status == ExecutionStatus.CLOSED:
            await self._on_close(
                execution.market_id,
                execution.thesis_id,
                execution,
            )

    async def stop(self) -> None:
        self._running = False
        logger.info("ExecutionSubscriber: stopped")
