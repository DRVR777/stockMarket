"""Step 5 — Cascade detection.

Maintains Redis sorted sets keyed by ``oracle:state:cascade:{market_id}:{outcome}``
to track distinct wallets that filled within a time window. If >= 3 distinct
wallets appear in the last 300 seconds, a cascade_buy event is flagged.

Each sorted set has a TTL of 600 seconds to auto-expire stale data.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from whale_detector.config import (
    CASCADE_KEY_PREFIX,
    CASCADE_MIN_WALLETS,
    CASCADE_SET_TTL_SECONDS,
    CASCADE_WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)


class CascadeDetector:
    """Detect coordinated multi-wallet activity on a market+outcome pair."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def check(
        self,
        market_id: str,
        outcome: str,
        wallet_address: str,
        event_timestamp: Optional[float] = None,
    ) -> tuple[bool, list[str]]:
        """Record a wallet fill and check for cascade activity.

        Parameters
        ----------
        market_id:
            The market being traded.
        outcome:
            The outcome side (e.g. YES token id).
        wallet_address:
            The wallet that placed the fill.
        event_timestamp:
            Unix timestamp of the event. Defaults to ``time.time()``.

        Returns
        -------
        tuple[bool, list[str]]
            ``(is_cascade, cascade_wallets)`` — True if >= CASCADE_MIN_WALLETS
            distinct wallets within CASCADE_WINDOW_SECONDS. The wallet list
            contains all distinct wallets in the window.
        """
        now = event_timestamp or time.time()
        key = f"{CASCADE_KEY_PREFIX}:{market_id}:{outcome}"

        # Add this wallet to the sorted set (score = timestamp).
        # If wallet already exists, the score is updated to the latest time.
        await self._redis.zadd(key, {wallet_address: now})

        # Set TTL on the sorted set so it auto-expires
        await self._redis.expire(key, CASCADE_SET_TTL_SECONDS)

        # Query all entries within the cascade window
        window_start = now - CASCADE_WINDOW_SECONDS
        recent_wallets: list[str] = await self._redis.zrangebyscore(
            key, window_start, now,
        )

        # Deduplicate (Redis sorted sets are inherently unique by member,
        # but defensive coding for any edge case)
        distinct = list(set(recent_wallets))

        is_cascade = len(distinct) >= CASCADE_MIN_WALLETS

        if is_cascade:
            logger.info(
                "CascadeDetector: CASCADE detected on %s:%s — %d wallets in %ds",
                market_id,
                outcome,
                len(distinct),
                CASCADE_WINDOW_SECONDS,
            )

        return is_cascade, distinct
