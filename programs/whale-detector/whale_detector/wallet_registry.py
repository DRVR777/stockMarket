"""Steps 3 & 8 — Wallet registry lookup and update.

Manages WalletProfile objects stored in Redis at ``oracle:state:wallets``.
On each flagged event:
  - Looks up the wallet; creates a stub if not found (Step 3).
  - After processing, recalculates ``typical_position_size_usd`` (rolling
    median of last 20 fills) and updates ``reputation_tier`` (Step 8).
"""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from typing import Any, Optional

from oracle_shared.contracts.wallet_profile import (
    INFORMED_POS_SIZE,
    INFORMED_WIN_RATE,
    MIN_TRADES_TO_CLASSIFY,
    NOISE_WIN_RATE,
    SHARK_POS_SIZE,
    SHARK_WIN_RATE,
    ReputationTier,
    WalletProfile,
)

from whale_detector.config import (
    WALLET_FILLS_KEY_PREFIX,
    WALLET_FILLS_MAX_LENGTH,
)

logger = logging.getLogger(__name__)


class WalletRegistry:
    """Read, create, and update WalletProfile objects in Redis."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def get_or_create(self, wallet_address: str) -> WalletProfile:
        """Fetch an existing WalletProfile or create a stub with tier=Unknown."""
        raw: Optional[str] = await self._redis.hget(
            WalletProfile.STATE_KEY_PREFIX, wallet_address,
        )
        if raw is not None:
            try:
                return WalletProfile.model_validate_json(raw)
            except Exception:
                logger.warning(
                    "WalletRegistry: corrupt profile for %s — recreating",
                    wallet_address,
                )

        now = datetime.now(timezone.utc)
        profile = WalletProfile(
            wallet_address=wallet_address,
            reputation_tier=ReputationTier.UNKNOWN,
            first_seen_at=now,
            last_active_at=now,
        )
        await self._save(profile)
        logger.info("WalletRegistry: created stub profile for %s", wallet_address)
        return profile

    async def update_after_event(
        self,
        profile: WalletProfile,
        size_usd: float,
    ) -> WalletProfile:
        """Recalculate rolling stats and reputation tier after an event (Step 8).

        Uses a Redis pipeline (MULTI/EXEC) for atomic fill list + profile update.
        """
        fills_key = f"{WALLET_FILLS_KEY_PREFIX}:{profile.wallet_address}"

        # Atomic: append fill, trim, and read back in one pipeline
        async with self._redis.pipeline(transaction=True) as pipe:
            await pipe.rpush(fills_key, str(size_usd))
            await pipe.ltrim(fills_key, -WALLET_FILLS_MAX_LENGTH, -1)
            await pipe.lrange(fills_key, 0, -1)
            results = await pipe.execute()

        raw_fills = results[2]  # lrange result is the 3rd command
        fills = [float(f) for f in raw_fills]

        profile.typical_position_size_usd = (
            statistics.median(fills) if fills else 0.0
        )
        profile.total_trades_tracked += 1
        profile.last_active_at = datetime.now(timezone.utc)

        # Recalculate reputation tier
        profile.reputation_tier = self._compute_tier(profile)

        await self._save(profile)
        return profile

    @staticmethod
    def _compute_tier(profile: WalletProfile) -> ReputationTier:
        """Algorithmic tier assignment based on thresholds in wallet_profile.py.

        Requires at least ``MIN_TRADES_TO_CLASSIFY`` trades; otherwise remains
        Unknown. Tier is determined by win_rate and typical_position_size_usd.
        """
        if profile.total_trades_tracked < MIN_TRADES_TO_CLASSIFY:
            return ReputationTier.UNKNOWN

        wr = profile.win_rate
        pos = profile.typical_position_size_usd

        if wr >= SHARK_WIN_RATE and pos >= SHARK_POS_SIZE:
            return ReputationTier.SHARK
        if wr >= INFORMED_WIN_RATE or pos >= INFORMED_POS_SIZE:
            return ReputationTier.INFORMED
        if wr < NOISE_WIN_RATE:
            return ReputationTier.NOISE

        return ReputationTier.UNKNOWN

    async def _save(self, profile: WalletProfile) -> None:
        """Persist a WalletProfile to Redis."""
        await self._redis.hset(
            WalletProfile.STATE_KEY_PREFIX,
            profile.wallet_address,
            profile.model_dump_json(),
        )
