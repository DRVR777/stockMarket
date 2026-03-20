"""Task 8 — Scheduled full market scan.

APScheduler fires every N minutes (default 30). For each market in
oracle:state:markets, runs the full pipeline (Steps 1-4 → emit thesis).
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from reasoning_engine.config import (
    MARKET_STATE_KEY,
    PARAMS_KEY,
    RE_SCAN_INTERVAL_DEFAULT,
    RE_SCAN_INTERVAL_PARAM,
)

logger = logging.getLogger(__name__)

# Callback: async def analyze_market(market_id: str) -> None
AnalyzeCallback = Callable[[str], Awaitable[None]]


class ScheduledScanner:
    """Run a full market scan on a configurable interval."""

    def __init__(
        self,
        redis_client: Any,
        analyze_market: AnalyzeCallback,
    ) -> None:
        self._redis = redis_client
        self._analyze = analyze_market
        self._scheduler = AsyncIOScheduler()

    async def start(self) -> None:
        interval = await self._get_interval()
        self._scheduler.add_job(
            self._scan,
            "interval",
            minutes=interval,
            id="re_full_scan",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "ScheduledScanner: started  interval=%d minutes", interval
        )

    async def _scan(self) -> None:
        """Scan all active markets and trigger analysis."""
        market_ids = await self._redis.hkeys(MARKET_STATE_KEY)
        logger.info("ScheduledScanner: full scan  markets=%d", len(market_ids))

        for market_id in market_ids:
            try:
                await self._analyze(market_id)
            except Exception:
                logger.exception(
                    "ScheduledScanner: failed to analyze market %s", market_id
                )

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("ScheduledScanner: stopped")

    async def _get_interval(self) -> int:
        raw = await self._redis.hget(PARAMS_KEY, RE_SCAN_INTERVAL_PARAM)
        if raw:
            try:
                return int(raw)
            except (ValueError, TypeError):
                pass
        return RE_SCAN_INTERVAL_DEFAULT
