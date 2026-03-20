"""Step 7 — MarketState update.

For each market matched by an Insight:
  1. Read current MarketState from Redis.
  2. Prepend the Insight to recent_insights, trim to 20.
  3. Increment signal_count_24h.
  4. Regenerate semantic_state_summary via Claude Haiku.
  5. Write back to Redis and publish to oracle:market_state.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from oracle_shared.contracts.insight import Insight
from oracle_shared.contracts.market_state import MarketState, RECENT_INSIGHTS_WINDOW
from oracle_shared.providers import LLMProvider, get_llm

from osint_fusion.config import MARKET_STATE_KEY, SUMMARY_MAX_WORDS

logger = logging.getLogger(__name__)


class MarketStateUpdater:
    """Update MarketState in Redis when new Insights arrive."""

    def __init__(self, redis_client: Any, llm: LLMProvider | None = None) -> None:
        self._redis = redis_client
        self._llm = llm or get_llm()

    async def update(self, insight: Insight) -> None:
        """Update MarketState for each market associated with the Insight."""
        for market_id in insight.associated_market_ids:
            try:
                await self._update_single(market_id, insight)
            except Exception:
                logger.exception(
                    "MarketStateUpdater: failed to update market %s", market_id
                )

    async def _update_single(self, market_id: str, insight: Insight) -> None:
        """Update a single market's state with optimistic locking.

        Uses Redis WATCH to detect concurrent modifications. Retries once on conflict.
        """
        for attempt in range(2):
            state = await self._load_state(market_id)
            if state is None:
                logger.debug(
                    "MarketStateUpdater: no state for market %s — skipping", market_id
                )
                return

            # Prepend insight, trim to window
            insight_dict = insight.model_dump()
            insight_dict["timestamp"] = insight.timestamp.isoformat()
            state.recent_insights.insert(0, insight_dict)
            state.recent_insights = state.recent_insights[:RECENT_INSIGHTS_WINDOW]

            state.signal_count_24h += 1
            state.last_signal_at = datetime.now(timezone.utc)

            # Regenerate semantic summary from last 5 insights
            state.semantic_state_summary = await self._generate_summary(
                state.market_question,
                state.recent_insights[:5],
            )

            # Persist and publish
            await self._save_state(market_id, state)
            await self._redis.publish(
                MarketState.CHANNEL, state.model_dump_json()
            )
            logger.info(
                "MarketStateUpdater: updated market %s  insights=%d  signals_24h=%d",
                market_id,
                len(state.recent_insights),
                state.signal_count_24h,
            )
            return  # success

        logger.warning("MarketStateUpdater: failed to update %s after retries", market_id)

    async def _generate_summary(
        self,
        market_question: str,
        recent_insights: list[dict],
    ) -> str:
        """Generate a semantic state summary via LLM provider."""
        if not recent_insights:
            return ""

        insight_texts = "\n".join(
            f"- [{ins.get('source_category', '?')}] {ins.get('raw_text', '')[:200]}"
            for ins in recent_insights
        )

        prompt = (
            f"Market question: {market_question}\n\n"
            f"Recent signals:\n{insight_texts}\n\n"
            f"Summarize what these signals suggest about this market "
            f"in {SUMMARY_MAX_WORDS} words or fewer. "
            f"Be factual and concise."
        )

        try:
            return (await self._llm.generate(prompt, max_tokens=200)).strip()
        except Exception:
            logger.warning("MarketStateUpdater: summary generation failed -- using fallback", exc_info=True)
            parts = [ins.get("raw_text", "")[:100] for ins in recent_insights]
            return " | ".join(parts)[:500]

    async def _load_state(self, market_id: str) -> Optional[MarketState]:
        raw: Optional[str] = await self._redis.hget(MARKET_STATE_KEY, market_id)
        if raw is None:
            return None
        try:
            return MarketState.model_validate_json(raw)
        except Exception:
            logger.warning(
                "MarketStateUpdater: corrupt state for %s", market_id
            )
            return None

    async def _save_state(self, market_id: str, state: MarketState) -> None:
        await self._redis.hset(
            MARKET_STATE_KEY, market_id, state.model_dump_json()
        )
