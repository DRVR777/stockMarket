"""
Adapter 4a: NewsAPI
Polls GET /v2/everything every 5 minutes with market-derived keywords.
Emits Signal(source_id=newsapi, category=news) per article.
Publishes to redis channel: oracle:signal
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"


class NewsAPIAdapter:
    """
    Shared adapter interface:
        start()  — begin polling loop (runs until stop())
        stop()   — signal the loop to exit
    """

    def __init__(
        self,
        redis_client: Any,
        api_key: str | None = None,
        poll_interval: int = 300,
        default_keywords: list[str] | None = None,
    ) -> None:
        self._redis = redis_client
        self._api_key = api_key or os.getenv("NEWSAPI_API_KEY", "")
        self._poll_interval = poll_interval
        self._running = False
        self._default_keywords = default_keywords or [
            "prediction market",
            "Polymarket",
            "election odds",
            "sports betting odds",
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        if not self._api_key:
            logger.error("NewsAPIAdapter: NEWSAPI_API_KEY not set — skipping")
            return
        logger.info(
            "NewsAPIAdapter started  poll_interval=%ds  keywords=%d",
            self._poll_interval,
            len(self._default_keywords),
        )
        while self._running:
            try:
                await self._poll()
            except Exception:
                logger.exception("NewsAPIAdapter: unhandled error in poll()")
            if self._running:
                await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("NewsAPIAdapter stopped")

    # ── Core poll ─────────────────────────────────────────────────────────────

    async def _poll(self) -> None:
        query = " OR ".join(f'"{kw}"' for kw in self._default_keywords)
        published = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{NEWSAPI_BASE}/everything",
                params={
                    "q": query,
                    "sortBy": "publishedAt",
                    "pageSize": 50,
                    "language": "en",
                    "apiKey": self._api_key,
                },
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning("NewsAPIAdapter: rate limited — sleeping %ds", retry_after)
                await asyncio.sleep(retry_after)
                return
            if resp.status_code == 401:
                logger.error("NewsAPIAdapter: API key rejected (401) — pausing 1h")
                await asyncio.sleep(3600)
                return
            resp.raise_for_status()
            body = resp.json()

            for article in body.get("articles", []):
                signal = self._normalize(article, query)
                await self._publish(signal)
                published += 1

        logger.info("NewsAPIAdapter: published %d signals", published)

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize(self, article: dict[str, Any], query_used: str) -> Signal:
        """
        raw_payload keys (per signal.md):
            title, description, url, source_name, published_at, query_used
        """
        return Signal(
            source_id=SourceId.NEWSAPI,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.NEWS,
            raw_payload={
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "source_name": (article.get("source") or {}).get("name", ""),
                "published_at": article.get("publishedAt", ""),
                "query_used": query_used,
            },
        )

    # ── Publish ───────────────────────────────────────────────────────────────

    async def _publish(self, signal: Signal) -> None:
        """Publish a normalized Signal to the canonical Redis channel."""
        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())
