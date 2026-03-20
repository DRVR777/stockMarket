"""Step 5 — Source credibility weighting.

Reads per-category credibility weights from Redis, falling back to defaults
from the Insight contract. Used to weight how much a signal from a given
source should influence market analysis.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from oracle_shared.contracts.insight import DEFAULT_CREDIBILITY_WEIGHTS

from osint_fusion.config import (
    CREDIBILITY_PARAMS_PREFIX,
    CREDIBILITY_WEIGHT_MAX,
    CREDIBILITY_WEIGHT_MIN,
)

logger = logging.getLogger(__name__)


class CredibilityWeighter:
    """Look up and apply source credibility weights."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def get_weight(self, category: str) -> float:
        """Return the credibility weight for a signal category.

        Checks ``oracle:state:params:credibility_weights:{category}`` in Redis
        first, then falls back to the default from the Insight contract.
        """
        key = f"{CREDIBILITY_PARAMS_PREFIX}:{category}"
        raw: Optional[str] = await self._redis.get(key)
        if raw is not None:
            try:
                return self._clamp(float(raw))
            except (ValueError, TypeError):
                pass

        return DEFAULT_CREDIBILITY_WEIGHTS.get(category, 1.0)

    async def apply_delta(self, category: str, delta: float) -> float:
        """Apply a delta to a credibility weight (from PostMortem feedback).

        Returns the new clamped weight.
        """
        current = await self.get_weight(category)
        new_weight = self._clamp(current + delta)
        key = f"{CREDIBILITY_PARAMS_PREFIX}:{category}"
        await self._redis.set(key, str(new_weight))
        logger.info(
            "CredibilityWeighter: %s weight %.2f → %.2f (delta %+.3f)",
            category, current, new_weight, delta,
        )
        return new_weight

    @staticmethod
    def _clamp(value: float) -> float:
        return max(CREDIBILITY_WEIGHT_MIN, min(CREDIBILITY_WEIGHT_MAX, value))
