"""Step 2 — Threshold flagging.

Reads the operator-configurable ``large_order_threshold_usd`` from Redis
(``oracle:state:params``). If a signal's ``raw_payload.size_usd`` meets or
exceeds the threshold, it is flagged as a Large Order Event and forwarded
to the anomaly scoring pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from oracle_shared.contracts.signal import Signal

from whale_detector.config import (
    LARGE_ORDER_THRESHOLD_USD_DEFAULT,
    LARGE_ORDER_THRESHOLD_PARAM_KEY,
    PARAMS_STATE_KEY,
)

logger = logging.getLogger(__name__)


class ThresholdFlagger:
    """Evaluate incoming signals against the large-order USD threshold."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def get_threshold(self) -> float:
        """Read the current threshold from Redis, falling back to the default."""
        raw: Optional[str] = await self._redis.hget(
            PARAMS_STATE_KEY, LARGE_ORDER_THRESHOLD_PARAM_KEY,
        )
        if raw is not None:
            try:
                return float(raw)
            except (ValueError, TypeError):
                logger.warning(
                    "ThresholdFlagger: invalid threshold value in Redis: %r — "
                    "using default %.0f",
                    raw,
                    LARGE_ORDER_THRESHOLD_USD_DEFAULT,
                )
        return LARGE_ORDER_THRESHOLD_USD_DEFAULT

    async def evaluate(self, signal: Signal) -> bool:
        """Return True if the signal exceeds the large-order threshold.

        Expects ``raw_payload`` to contain a ``size_usd`` key (float).
        Signals missing this field are silently skipped (return False).
        """
        size_usd: float = signal.raw_payload.get("size_usd", 0.0)
        if not isinstance(size_usd, (int, float)):
            logger.debug(
                "ThresholdFlagger: signal %s missing numeric size_usd — skipping",
                signal.signal_id,
            )
            return False

        threshold = await self.get_threshold()
        flagged = size_usd >= threshold
        if flagged:
            logger.info(
                "ThresholdFlagger: FLAGGED signal %s  size_usd=%.2f >= threshold=%.2f",
                signal.signal_id,
                size_usd,
                threshold,
            )
        return flagged
