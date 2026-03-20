"""Task 3 — Step 1: Context assembly.

For a given market_id:
  (a) Read MarketState from Redis
  (b) Read last 5 AnomalyEvents from Redis anomaly index
  (c) Query oracle_theses ChromaDB for historical analogues
  (d) Assemble into ContextAssembly
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from oracle_shared.contracts.market_state import MarketState
from oracle_shared.contracts.trade_thesis import ContextAssembly, HistoricalAnalogue

from reasoning_engine.chroma_store import ThesisChromaStore
from reasoning_engine.config import ANOMALY_INDEX_PREFIX, MARKET_STATE_KEY

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Gather all context for a single market's analysis."""

    def __init__(
        self,
        redis_client: Any,
        chroma: ThesisChromaStore,
        embed_fn: Any,
    ) -> None:
        self._redis = redis_client
        self._chroma = chroma
        self._embed_fn = embed_fn  # async callable: str -> list[float]

    async def assemble(self, market_id: str) -> Optional[ContextAssembly]:
        """Build a ContextAssembly for the given market.

        Returns None if the market doesn't exist in Redis state.
        """
        # (a) Read MarketState
        market_state = await self._load_market_state(market_id)
        if market_state is None:
            logger.debug("ContextAssembler: no state for market %s", market_id)
            return None

        # (b) Read last 5 AnomalyEvents
        anomaly_events = await self._load_anomaly_events(market_id, limit=5)

        # (c) Historical analogues via ChromaDB
        analogues = await self._find_analogues(market_state.market_question)

        assembly = ContextAssembly(
            market_state=market_state.model_dump(),
            anomaly_events=anomaly_events,
            historical_analogues=analogues,
            assembled_at=datetime.now(timezone.utc),
        )

        logger.info(
            "ContextAssembler: assembled for %s  anomalies=%d  analogues=%d",
            market_id,
            len(anomaly_events),
            len(analogues),
        )
        return assembly

    async def _load_market_state(self, market_id: str) -> Optional[MarketState]:
        raw = await self._redis.hget(MARKET_STATE_KEY, market_id)
        if raw is None:
            return None
        try:
            return MarketState.model_validate_json(raw)
        except Exception:
            logger.warning("ContextAssembler: corrupt market state for %s", market_id)
            return None

    async def _load_anomaly_events(
        self, market_id: str, limit: int = 5,
    ) -> list[dict]:
        key = f"{ANOMALY_INDEX_PREFIX}:{market_id}"
        raw_list = await self._redis.lrange(key, 0, limit - 1)
        events = []
        for raw in raw_list:
            try:
                events.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        return events

    async def _find_analogues(
        self, market_question: str,
    ) -> list[HistoricalAnalogue]:
        try:
            embedding = await self._embed_fn(market_question)
            return self._chroma.find_analogues(embedding)
        except Exception:
            logger.warning("ContextAssembler: analogue search failed", exc_info=True)
            return []
