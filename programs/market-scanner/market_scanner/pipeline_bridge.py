"""Bridge between market-scanner and the ORACLE pipeline.

Converts high-confidence SMC/pattern results into Signal objects
and publishes them to oracle:signal for RE to pick up. Also persists
scanner results to Postgres.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId
from oracle_shared.db import get_session
from oracle_shared.db.models import SignalRow

from market_scanner.patterns import PatternResult
from market_scanner.smc import SMCAnalysis, Bias

logger = logging.getLogger(__name__)


class PipelineBridge:
    """Convert scanner results to ORACLE Signals and publish."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def publish_pattern(self, result: PatternResult) -> None:
        """Convert a PatternResult to a Signal and publish."""
        signal = Signal(
            source_id=SourceId.AI_OPINION,  # closest category for scanner output
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.PRICE,
            raw_payload={
                "scanner_type": "technical_pattern",
                "symbol": result.symbol,
                "asset_type": result.asset_type,
                "direction": result.direction.value,
                "score": result.score,
                "patterns": result.patterns,
                "current_price": result.current_price,
                "rsi": result.rsi,
                "macd_signal": result.macd_signal,
                "ma_trend": result.ma_trend,
                "volume_ratio": result.volume_ratio,
                "bb_squeeze": result.bb_squeeze,
                "support": result.support,
                "resistance": result.resistance,
                "entry_price": result.entry_price,
                "stop_loss": result.stop_loss,
                "take_profit": result.take_profit,
            },
            confidence=result.score,
        )

        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())
        await self._persist(signal)

        logger.info(
            "PipelineBridge: published pattern signal %s %s score=%.2f",
            result.symbol, result.direction.value, result.score,
        )

    async def publish_smc(self, analysis: SMCAnalysis) -> None:
        """Convert an SMC analysis to a Signal and publish."""
        signal = Signal(
            source_id=SourceId.AI_OPINION,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.PRICE,
            raw_payload={
                "scanner_type": "smc_analysis",
                "symbol": analysis.symbol,
                "asset_type": analysis.asset_type,
                "bias": analysis.bias.value,
                "confidence": analysis.confidence,
                "trend": analysis.trend,
                "setup_type": analysis.setup_type,
                "signals": analysis.signals,
                "current_price": analysis.current_price,
                "equilibrium": analysis.equilibrium,
                "premium_zone": list(analysis.premium_zone),
                "discount_zone": list(analysis.discount_zone),
                "entry_zone": list(analysis.entry_zone) if analysis.entry_zone else None,
                "stop_loss": analysis.stop_loss,
                "take_profit": analysis.take_profit,
                "fvg_count": len(analysis.fair_value_gaps),
                "ob_count": len(analysis.order_blocks),
                "liquidity_levels": len(analysis.liquidity_levels),
                "structure_breaks": [
                    {"type": b.type, "price": b.price}
                    for b in analysis.structure_breaks
                ],
            },
            confidence=analysis.confidence,
        )

        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())
        await self._persist(signal)

        logger.info(
            "PipelineBridge: published SMC signal %s %s conf=%.2f setup=%s",
            analysis.symbol, analysis.bias.value, analysis.confidence,
            analysis.setup_type or "none",
        )

    async def _persist(self, signal: Signal) -> None:
        """Persist signal to Postgres."""
        try:
            async with get_session() as session:
                session.add(SignalRow.from_contract(signal))
        except Exception:
            logger.debug("PipelineBridge: Postgres persist failed", exc_info=True)
