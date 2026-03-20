"""Step 4 — Anomaly scoring.

Computes a 0.0-1.0 anomaly score as a weighted sum of three equally-weighted
factors:

  (a) **Size ratio** — ``size_usd / market_liquidity_usd``, clamped [0, 1].
  (b) **Wallet ratio** — ``size_usd / wallet.typical_position_size_usd``,
      normalized so that >= 2x the typical size scores 1.0.
  (c) **Time ratio** — ``1 - (hours_to_resolution / 168)``, clamped [0, 1].
      168 hours = 7 days. Markets resolving sooner score higher.

Market liquidity is read from ``oracle:state:markets:{market_id}``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from oracle_shared.contracts.wallet_profile import WalletProfile

from whale_detector.config import (
    ANOMALY_WEIGHT_SIZE,
    ANOMALY_WEIGHT_TIME,
    ANOMALY_WEIGHT_WALLET,
    MARKET_STATE_KEY,
    RESOLUTION_HORIZON_HOURS,
    WALLET_POSITION_MULTIPLIER_CAP,
)

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """Score a flagged on-chain event from three dimensions."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def score(
        self,
        size_usd: float,
        market_id: str,
        wallet_profile: WalletProfile,
    ) -> tuple[float, list[str]]:
        """Return (anomaly_score, trigger_reasons) for a flagged signal.

        Reads market liquidity and end_date from Redis state to compute
        the size and time factors. Returns both the composite score and
        a list of trigger_reason tags.
        """
        trigger_reasons: list[str] = ["large_order"]

        size_factor = await self._size_factor(size_usd, market_id)

        wallet_factor = self._wallet_factor(size_usd, wallet_profile)
        if wallet_profile.typical_position_size_usd > 0:
            raw_ratio = size_usd / wallet_profile.typical_position_size_usd
            if raw_ratio >= WALLET_POSITION_MULTIPLIER_CAP:
                trigger_reasons.append("size_deviation")

        hours_to_res = await self._get_hours_to_resolution(market_id)
        time_factor = self._time_factor(hours_to_res)
        if hours_to_res is not None and hours_to_res <= RESOLUTION_HORIZON_HOURS:
            trigger_reasons.append("pre_resolution")

        # Wallet tier triggers
        tier = wallet_profile.reputation_tier.value
        if tier == "Shark":
            trigger_reasons.append("tier_1_wallet")
        elif tier == "Informed":
            trigger_reasons.append("tier_2_wallet")

        composite = (
            ANOMALY_WEIGHT_SIZE * size_factor
            + ANOMALY_WEIGHT_WALLET * wallet_factor
            + ANOMALY_WEIGHT_TIME * time_factor
        )
        composite = max(0.0, min(1.0, composite))

        logger.info(
            "AnomalyScorer: score=%.3f  factors=(%.2f, %.2f, %.2f)  reasons=%s",
            composite, size_factor, wallet_factor, time_factor, trigger_reasons,
        )
        return round(composite, 4), trigger_reasons

    # -- Factor (a): size_usd / market_liquidity_usd -------------------------

    async def _size_factor(self, size_usd: float, market_id: str) -> float:
        """Ratio of fill size to market liquidity, clamped [0, 1]."""
        liquidity = await self._get_market_liquidity(market_id)
        if liquidity <= 0:
            # No liquidity data available — assume moderate impact
            return 0.5
        ratio = size_usd / liquidity
        return max(0.0, min(1.0, ratio))

    async def _get_market_liquidity(self, market_id: str) -> float:
        """Read ``liquidity_usd`` from the MarketState Redis hash."""
        raw: Optional[str] = await self._redis.hget(
            MARKET_STATE_KEY, market_id,
        )
        if raw is None:
            return 0.0
        try:
            market_data = json.loads(raw)
            # MarketState contract canonical key is "liquidity_usd"
            return float(market_data.get("liquidity_usd", 0.0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0.0

    async def _get_hours_to_resolution(self, market_id: str) -> Optional[float]:
        """Read market end_date from Redis state and compute hours remaining."""
        raw: Optional[str] = await self._redis.hget(MARKET_STATE_KEY, market_id)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            end_date_str = data.get("end_date", "")
            if not end_date_str:
                return None
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            delta = end_dt - datetime.now(timezone.utc)
            return max(0.0, delta.total_seconds() / 3600.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    # -- Factor (b): size_usd / typical_position_size_usd --------------------

    @staticmethod
    def _wallet_factor(size_usd: float, profile: WalletProfile) -> float:
        """Ratio of fill size to wallet's typical position, normalized.

        If the wallet has no history (typical = 0), return 1.0 (maximum
        surprise). A ratio >= ``WALLET_POSITION_MULTIPLIER_CAP`` (default 2x)
        also scores 1.0.
        """
        typical = profile.typical_position_size_usd
        if typical <= 0:
            return 1.0
        ratio = size_usd / typical
        normalized = ratio / WALLET_POSITION_MULTIPLIER_CAP
        return max(0.0, min(1.0, normalized))

    # -- Factor (c): 1 - (hours_to_resolution / 168) -------------------------

    @staticmethod
    def _time_factor(hours_to_resolution: Optional[float]) -> float:
        """Urgency score: markets resolving sooner score higher.

        If hours_to_resolution is None (unknown), return 0.5 (neutral).
        """
        if hours_to_resolution is None:
            return 0.5
        ratio = hours_to_resolution / RESOLUTION_HORIZON_HOURS
        return max(0.0, min(1.0, 1.0 - ratio))
