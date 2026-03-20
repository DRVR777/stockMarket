"""Task 7 — TradeThesis emission.

Assembles the full TradeThesis from pipeline outputs, publishes to Redis,
indexes in ChromaDB for future analogue search, and persists to Postgres.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from oracle_shared.contracts.trade_thesis import (
    ContextAssembly,
    EvidenceWeight,
    Hypothesis,
    ThesisDecision,
    TradeThesis,
)
from oracle_shared.db import get_session
from oracle_shared.db.repository import ThesisRepo
from oracle_shared.providers import EmbeddingProvider, get_embedder

from reasoning_engine.chroma_store import ThesisChromaStore
from reasoning_engine.config import THESES_INDEX_KEY

logger = logging.getLogger(__name__)


class ThesisEmitter:
    """Assemble, publish, index, and persist TradeThesis objects."""

    def __init__(
        self,
        redis_client: Any,
        chroma: ThesisChromaStore,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._redis = redis_client
        self._chroma = chroma
        self._embedder = embedder or get_embedder()

    async def emit(
        self,
        market_id: str,
        market_question: str,
        hypotheses: list[Hypothesis],
        evidence_weights: list[EvidenceWeight],
        context: ContextAssembly,
        re_probability: float,
        probability_delta: float,
        confidence: float,
        decision: ThesisDecision,
        position_usd: Optional[float],
    ) -> TradeThesis:
        """Build, publish, index, and persist a TradeThesis."""
        # Determine direction
        direction = "YES" if re_probability >= 0.5 else "NO"

        thesis = TradeThesis(
            created_at=datetime.now(timezone.utc),
            market_id=market_id,
            market_question=market_question,
            direction=direction,
            re_probability_estimate=re_probability,
            market_implied_probability=float(
                context.market_state.get("current_price_yes", 0.5)
            ),
            probability_delta=probability_delta,
            confidence_score=confidence,
            decision=decision,
            recommended_position_usd=position_usd,
            hypotheses=hypotheses,
            evidence_weights=evidence_weights,
            context_assembly=context,
        )

        # Publish to Redis
        await self._redis.publish(TradeThesis.CHANNEL, thesis.model_dump_json())

        # Index thesis_id in Redis set
        await self._redis.sadd(THESES_INDEX_KEY, thesis.thesis_id)

        # Index in ChromaDB for future analogue search
        try:
            embedding = await self._embed(market_question)
            self._chroma.upsert(
                thesis_id=thesis.thesis_id,
                market_question=market_question,
                embedding=embedding,
                metadata={
                    "market_id": market_id,
                    "direction": direction,
                    "decision": decision.value,
                    "confidence": confidence,
                },
            )
        except Exception:
            logger.warning("ThesisEmitter: ChromaDB index failed", exc_info=True)

        # Persist to Postgres
        try:
            async with get_session() as session:
                await ThesisRepo.save(session, thesis)
        except Exception:
            logger.warning("ThesisEmitter: Postgres save failed", exc_info=True)

        logger.info(
            "ThesisEmitter: published %s  market=%s  direction=%s  "
            "decision=%s  confidence=%.3f  delta=%.3f",
            thesis.thesis_id,
            market_id[:16],
            direction,
            decision.value,
            confidence,
            probability_delta,
        )
        return thesis

    async def _embed(self, text: str) -> list[float]:
        return await self._embedder.embed_single(text)
