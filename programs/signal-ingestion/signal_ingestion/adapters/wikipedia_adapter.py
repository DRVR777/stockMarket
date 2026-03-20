"""
Adapter 4b: Wikipedia Recent Changes
Polls the MediaWiki Recent Changes API every 15 minutes for edits to pages
matching active market topics.
Emits Signal(source_id=wikipedia, category=news) per relevant edit.
Publishes to redis channel: oracle:signal
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

# MediaWiki Action API — no API key required
MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"


class WikipediaAdapter:
    """
    Shared adapter interface:
        start()  — begin polling loop (runs until stop())
        stop()   — signal the loop to exit
    """

    def __init__(
        self,
        redis_client: Any,
        poll_interval: int = 900,
        watch_pages: list[str] | None = None,
    ) -> None:
        self._redis = redis_client
        self._poll_interval = poll_interval
        self._running = False
        self._watch_pages = watch_pages or [
            "United States presidential election",
            "Prediction market",
            "Polymarket",
            "2026 United States elections",
            "Super Bowl",
            "FIFA World Cup",
            "Federal Reserve",
            "Inflation",
        ]
        self._last_poll: datetime = datetime.now(timezone.utc) - timedelta(minutes=15)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info(
            "WikipediaAdapter started  poll_interval=%ds  watch_pages=%d",
            self._poll_interval,
            len(self._watch_pages),
        )
        while self._running:
            try:
                await self._poll()
            except Exception:
                logger.exception("WikipediaAdapter: unhandled error in poll()")
            if self._running:
                await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("WikipediaAdapter stopped")

    # ── Core poll ─────────────────────────────────────────────────────────────

    async def _poll(self) -> None:
        since = self._last_poll
        self._last_poll = datetime.now(timezone.utc)
        published = 0

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "oracle-sil/0.1 (https://github.com/oracle; signal-ingestion bot)"},
        ) as client:
            for page_title in self._watch_pages:
                edits = await self._fetch_recent_edits(client, page_title, since)
                for edit in edits:
                    signal = self._normalize(edit, page_title)
                    await self._publish(signal)
                    published += 1

        logger.info("WikipediaAdapter: published %d signals", published)

    async def _fetch_recent_edits(
        self,
        client: httpx.AsyncClient,
        page_title: str,
        since: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch revisions for a single page since the given timestamp."""
        params = {
            "action": "query",
            "titles": page_title,
            "prop": "revisions",
            "rvprop": "timestamp|comment|user|ids",
            "rvlimit": "10",
            "rvstart": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rvend": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rvdir": "older",
            "format": "json",
            "formatversion": "2",
        }

        resp = await client.get(MEDIAWIKI_API, params=params)
        resp.raise_for_status()
        body = resp.json()

        pages = body.get("query", {}).get("pages", [])
        if not pages:
            return []

        page = pages[0]
        if page.get("missing"):
            return []

        return page.get("revisions", [])

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize(self, edit: dict[str, Any], page_title: str) -> Signal:
        """
        raw_payload keys (per signal.md):
            page_title, summary, edit_timestamp, edit_comment, diff_url
        """
        rev_id = edit.get("revid", "")
        diff_url = (
            f"https://en.wikipedia.org/w/index.php?diff={rev_id}"
            if rev_id
            else ""
        )

        return Signal(
            source_id=SourceId.WIKIPEDIA,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.NEWS,
            raw_payload={
                "page_title": page_title,
                "summary": edit.get("comment", ""),
                "edit_timestamp": edit.get("timestamp", ""),
                "edit_comment": edit.get("comment", ""),
                "diff_url": diff_url,
            },
        )

    # ── Publish ───────────────────────────────────────────────────────────────

    async def _publish(self, signal: Signal) -> None:
        """Publish a normalized Signal to the canonical Redis channel."""
        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())
