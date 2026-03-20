"""Task 6 — Step 4: Confidence calibration.

Three sub-scores averaged:
  (a) recency:  fraction of recent_insights from last 6h vs total window
  (b) diversity: unique source categories / 5
  (c) model_certainty: 1 - normalized entropy of YES_score vs NO_score

If confidence < min threshold → decision=flag_for_review.
Else → decision=execute.

Computes recommended_position_usd when decision=execute.
"""
from __future__ import annotations

import math
import logging
from typing import Any, Optional
from datetime import datetime, timezone, timedelta

from oracle_shared.contracts.trade_thesis import (
    ContextAssembly,
    ThesisDecision,
)

from reasoning_engine.config import (
    BASE_STAKE_USD_DEFAULT,
    BASE_STAKE_USD_PARAM,
    MAX_POSITION_USD_DEFAULT,
    MAX_POSITION_USD_PARAM,
    PARAMS_KEY,
    RE_CONFIDENCE_MIN_DEFAULT,
    RE_CONFIDENCE_MIN_PARAM,
)

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Compute confidence score and determine execution decision."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def calibrate(
        self,
        context: ContextAssembly,
        re_probability: float,
        probability_delta: float,
    ) -> tuple[float, ThesisDecision, Optional[float]]:
        """Compute confidence and decide.

        Returns:
            (confidence_score, decision, recommended_position_usd)
        """
        ms = context.market_state
        insights = ms.get("recent_insights", [])

        # (a) Recency: fraction of insights from last 6 hours
        recency = self._recency_score(insights)

        # (b) Diversity: unique source categories / 5
        diversity = self._diversity_score(insights)

        # (c) Model certainty: 1 - normalized entropy
        certainty = self._certainty_score(re_probability)

        confidence = round((recency + diversity + certainty) / 3.0, 4)

        # Determine decision
        min_confidence = await self._get_min_confidence()
        if confidence < min_confidence:
            decision = ThesisDecision.FLAG_FOR_REVIEW
            position_usd = None
        else:
            decision = ThesisDecision.EXECUTE
            position_usd = await self._compute_position_size(
                confidence, probability_delta,
            )

        logger.info(
            "ConfidenceCalibrator: confidence=%.3f  "
            "(recency=%.2f diversity=%.2f certainty=%.2f)  "
            "decision=%s  position=$%s",
            confidence, recency, diversity, certainty,
            decision.value,
            f"{position_usd:,.0f}" if position_usd else "none",
        )

        return confidence, decision, position_usd

    @staticmethod
    def _recency_score(insights: list[dict]) -> float:
        """Fraction of insights from the last 6 hours."""
        if not insights:
            return 0.0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
        recent = 0
        for ins in insights:
            ts_str = ins.get("timestamp", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                if ts >= cutoff:
                    recent += 1
            except (ValueError, TypeError):
                continue
        return min(1.0, recent / max(len(insights), 1))

    @staticmethod
    def _diversity_score(insights: list[dict]) -> float:
        """Unique source categories / 5 (capped at 1.0)."""
        categories = {ins.get("source_category", "") for ins in insights}
        categories.discard("")
        return min(1.0, len(categories) / 5.0)

    @staticmethod
    def _certainty_score(re_probability: float) -> float:
        """1 - normalized entropy of the probability distribution.

        Binary entropy: H = -p*log2(p) - (1-p)*log2(1-p), max = 1.0 at p=0.5.
        Certainty = 1 - H.
        """
        p = max(0.001, min(0.999, re_probability))
        entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        return max(0.0, 1.0 - entropy)

    async def _get_min_confidence(self) -> float:
        raw = await self._redis.hget(PARAMS_KEY, RE_CONFIDENCE_MIN_PARAM)
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
        return RE_CONFIDENCE_MIN_DEFAULT

    async def _compute_position_size(
        self,
        confidence: float,
        probability_delta: float,
    ) -> float:
        """min(max_position, confidence * delta * base_stake)"""
        max_pos = await self._get_param(MAX_POSITION_USD_PARAM, MAX_POSITION_USD_DEFAULT)
        base_stake = await self._get_param(BASE_STAKE_USD_PARAM, BASE_STAKE_USD_DEFAULT)
        raw_size = confidence * abs(probability_delta) * base_stake
        return round(min(max_pos, raw_size), 2)

    async def _get_param(self, param_name: str, default: float) -> float:
        raw = await self._redis.hget(PARAMS_KEY, param_name)
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
        return default
