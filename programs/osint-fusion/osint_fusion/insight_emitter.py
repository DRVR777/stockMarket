"""Step 6 — Insight assembly and emission.

Assembles an Insight object from the pipeline outputs and publishes
to oracle:insight.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from oracle_shared.contracts.insight import Insight

logger = logging.getLogger(__name__)


class InsightEmitter:
    """Build and publish Insight objects."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def emit(
        self,
        source_signal_id: str,
        source_category: str,
        raw_text: str,
        similarity_scores: dict[str, float],
        credibility_weight: float,
    ) -> Insight:
        """Assemble and publish an Insight.

        ``semantic_summary`` is the first 500 chars of raw_text.
        Full summarization is deferred to the Reasoning Engine.
        """
        associated_market_ids = list(similarity_scores.keys())

        insight = Insight(
            timestamp=datetime.now(timezone.utc),
            source_signal_id=source_signal_id,
            source_category=source_category,
            associated_market_ids=associated_market_ids,
            similarity_scores=similarity_scores,
            semantic_summary=raw_text[:500],
            source_credibility_weight=credibility_weight,
            raw_text=raw_text,
        )

        await self._redis.publish(Insight.CHANNEL, insight.model_dump_json())
        logger.info(
            "InsightEmitter: published Insight %s  markets=%d  weight=%.2f",
            insight.insight_id,
            len(associated_market_ids),
            credibility_weight,
        )
        return insight
