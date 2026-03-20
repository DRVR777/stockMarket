"""
Adapter 4c: Reddit
Polls r/Polymarket, r/PredictIt, r/politics, r/sports every 10 minutes
via asyncpraw (async PRAW wrapper).
Emits Signal(source_id=reddit, category=social) per post.
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

REDDIT_OAUTH_URL = "https://oauth.reddit.com"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

DEFAULT_SUBREDDITS = ["Polymarket", "PredictIt", "politics", "sports"]


class RedditAdapter:
    """
    Shared adapter interface:
        start()  — begin polling loop (runs until stop())
        stop()   — signal the loop to exit

    Uses Reddit OAuth2 script-type app (client_credentials grant) via httpx
    to avoid PRAW's sync I/O and heavy dependency. Fetches /hot from each
    subreddit and deduplicates by post ID across cycles.
    """

    def __init__(
        self,
        redis_client: Any,
        poll_interval: int = 600,
        subreddits: list[str] | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self._redis = redis_client
        self._poll_interval = poll_interval
        self._subreddits = subreddits or DEFAULT_SUBREDDITS
        self._client_id = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        self._client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        self._running = False
        self._seen_ids: set[str] = set()
        self._access_token: str = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        if not self._client_id or not self._client_secret:
            logger.error(
                "RedditAdapter: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set — skipping"
            )
            return
        logger.info(
            "RedditAdapter started  poll_interval=%ds  subreddits=%s",
            self._poll_interval,
            self._subreddits,
        )
        while self._running:
            try:
                await self._poll()
            except Exception:
                logger.exception("RedditAdapter: unhandled error in poll()")
            if self._running:
                await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("RedditAdapter stopped")

    # ── OAuth token ───────────────────────────────────────────────────────────

    async def _ensure_token(self, client: httpx.AsyncClient) -> None:
        """Fetch or refresh Reddit OAuth2 bearer token (application-only)."""
        if self._access_token:
            return
        resp = await client.post(
            REDDIT_TOKEN_URL,
            auth=(self._client_id, self._client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": "oracle-sil/0.1"},
        )
        resp.raise_for_status()
        self._access_token = resp.json().get("access_token", "")

    # ── Core poll ─────────────────────────────────────────────────────────────

    async def _poll(self) -> None:
        published = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self._ensure_token(client)
            headers = {
                "Authorization": f"bearer {self._access_token}",
                "User-Agent": "oracle-sil/0.1",
            }

            for sub in self._subreddits:
                try:
                    posts = await self._fetch_subreddit(client, headers, sub)
                    for post in posts:
                        post_id = post.get("id", "")
                        if post_id in self._seen_ids:
                            continue
                        self._seen_ids.add(post_id)
                        signal = self._normalize(post, sub)
                        await self._publish(signal)
                        published += 1
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 401:
                        self._access_token = ""  # force re-auth next cycle
                        logger.warning("RedditAdapter: token expired — will refresh")
                    else:
                        logger.warning("RedditAdapter: HTTP %d for r/%s", exc.response.status_code, sub)

        # Cap seen set to prevent unbounded growth
        if len(self._seen_ids) > 10_000:
            self._seen_ids = set(list(self._seen_ids)[-5_000:])

        logger.info("RedditAdapter: published %d signals", published)

    async def _fetch_subreddit(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        subreddit: str,
    ) -> list[dict[str, Any]]:
        resp = await client.get(
            f"{REDDIT_OAUTH_URL}/r/{subreddit}/hot",
            headers=headers,
            params={"limit": "25", "raw_json": "1"},
        )
        resp.raise_for_status()
        body = resp.json()
        return [child["data"] for child in body.get("data", {}).get("children", [])]

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize(self, post: dict[str, Any], subreddit: str) -> Signal:
        """
        raw_payload keys (per signal.md):
            subreddit, post_id, title, score, num_comments, created_utc, url
        """
        return Signal(
            source_id=SourceId.REDDIT,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.SOCIAL,
            raw_payload={
                "subreddit": subreddit,
                "post_id": post.get("id", ""),
                "title": post.get("title", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "created_utc": post.get("created_utc", 0),
                "url": post.get("url", ""),
            },
        )

    # ── Publish ───────────────────────────────────────────────────────────────

    async def _publish(self, signal: Signal) -> None:
        """Publish a normalized Signal to the canonical Redis channel."""
        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())
