"""Data pipeline — ensures ALL signals get persisted to Postgres, embedded, and
prepared as ML features for the next model retraining.

This is the bridge between "collecting data" and "learning from data."

Every signal that flows through the daemon passes through here:
1. Persist to Postgres (permanent record)
2. Embed text data (news, Reddit, market questions) via Gemini
3. Store embeddings in Postgres for future ML training
4. Track outcomes for retraining dataset
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class DataPipeline:
    """Central data pipeline — persist, embed, prepare for ML."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._embedder = None

    async def _get_embedder(self):
        if self._embedder is None:
            try:
                from oracle_shared.providers import get_embedder
                self._embedder = get_embedder()
            except Exception:
                pass
        return self._embedder

    async def process_signal(self, signal_data: dict) -> None:
        """Process any signal: persist + embed + store.

        Call this for every signal the daemon generates, regardless of source.
        """
        signal_type = signal_data.get("asset_type", signal_data.get("setup_type", "unknown"))

        # 1. Persist to Postgres
        await self._persist_signal(signal_data)

        # 2. Embed any text content
        text_content = self._extract_text(signal_data)
        embedding = None
        if text_content:
            embedding = await self._embed(text_content)

        # 3. Store enriched record in Redis for fast access
        enriched = {
            **signal_data,
            "has_embedding": embedding is not None,
            "embedding_dims": len(embedding) if embedding else 0,
            "persisted_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._redis.lpush("oracle:enriched_signals", json.dumps(enriched, default=str))
        await self._redis.ltrim("oracle:enriched_signals", 0, 9999)

        # 4. Store embedding separately for ML training
        if embedding:
            embed_record = {
                "symbol": signal_data.get("symbol", ""),
                "direction": signal_data.get("direction", ""),
                "signal_type": signal_type,
                "embedding": embedding[:50],  # store truncated for Redis, full in Postgres
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self._redis.lpush("oracle:signal_embeddings", json.dumps(embed_record, default=str))
            await self._redis.ltrim("oracle:signal_embeddings", 0, 4999)

    async def _persist_signal(self, signal_data: dict) -> None:
        """Persist signal to Postgres."""
        try:
            from oracle_shared.db import get_session
            from oracle_shared.db.models import SignalRow
            from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

            # Map signal types to categories
            asset_type = signal_data.get("asset_type", "")
            if asset_type == "stock":
                category = SignalCategory.PRICE
            elif asset_type == "prediction_market":
                category = SignalCategory.AI
            elif asset_type == "crypto_funding":
                category = SignalCategory.PRICE
            elif asset_type == "crypto_cascade":
                category = SignalCategory.ON_CHAIN
            else:
                category = SignalCategory.PRICE

            sig = Signal(
                source_id=SourceId.AI_OPINION,
                timestamp=datetime.now(timezone.utc),
                category=category,
                raw_payload=signal_data,
                confidence=signal_data.get("ml_win_probability", signal_data.get("smc_confidence", 0)),
            )
            async with get_session() as session:
                session.add(SignalRow.from_contract(sig))

            logger.debug("DataPipeline: persisted %s %s to Postgres",
                         signal_data.get("symbol", "?"), signal_data.get("setup_type", "?"))
        except Exception:
            logger.debug("DataPipeline: Postgres persist failed", exc_info=True)

    def _extract_text(self, signal_data: dict) -> str:
        """Extract embeddable text from a signal."""
        parts = []

        # News headlines
        signals = signal_data.get("signals", [])
        if isinstance(signals, list):
            parts.extend(str(s) for s in signals[:3])

        # Polymarket question
        question = signal_data.get("question", "")
        if question:
            parts.append(question)

        # Symbol + direction as context
        symbol = signal_data.get("symbol", "")
        direction = signal_data.get("direction", "")
        setup = signal_data.get("setup_type", "")
        if symbol:
            parts.append(f"{symbol} {direction} {setup}")

        text = ". ".join(parts)
        return text if len(text) > 10 else ""

    async def _embed(self, text: str) -> list[float] | None:
        """Embed text using the configured provider."""
        embedder = await self._get_embedder()
        if embedder is None:
            return None
        try:
            return await embedder.embed_single(text)
        except Exception:
            return None

    async def process_outcome(self, trade_data: dict) -> None:
        """Process a resolved trade outcome — persist for retraining.

        This is the most valuable data: signal + features + actual outcome.
        Used by the weekly retraining job to improve the model.
        """
        try:
            from oracle_shared.db import get_session
            from oracle_shared.db.models import TradeExecutionRow
            from oracle_shared.contracts.trade_execution import (
                TradeExecution, MarketType, ExecutionSource, ExecutionStatus, ExitReason,
            )

            status = ExecutionStatus.CLOSED if trade_data.get("outcome") else ExecutionStatus.OPEN
            exit_reason = None
            if trade_data.get("close_reason") == "take_profit":
                exit_reason = ExitReason.TAKE_PROFIT
            elif trade_data.get("close_reason") == "stop_loss":
                exit_reason = ExitReason.STOP_LOSS

            # Determine market type from asset_type
            asset_type = trade_data.get("asset_type", "")
            if asset_type == "stock":
                market_type = MarketType.POLYMARKET  # closest for stocks
            elif asset_type == "prediction_market":
                market_type = MarketType.POLYMARKET
            else:
                market_type = MarketType.SOLANA  # crypto default

            execution = TradeExecution(
                market_id=trade_data.get("symbol", ""),
                market_type=market_type,
                direction=trade_data.get("direction", "buy"),
                entry_price=trade_data.get("entry_price", 0),
                size_usd=trade_data.get("size_usd", 0),
                executed_at=datetime.now(timezone.utc),
                execution_source=ExecutionSource.SOE_MEAN_REVERSION,
                status=status,
                exit_price=trade_data.get("exit_price"),
                exit_reason=exit_reason,
                realized_pnl_usd=trade_data.get("pnl_usd"),
            )

            async with get_session() as session:
                session.add(TradeExecutionRow.from_contract(execution))

            logger.debug("DataPipeline: persisted outcome %s %s pnl=$%.2f",
                         trade_data.get("symbol"), trade_data.get("outcome"),
                         trade_data.get("pnl_usd", 0))
        except Exception:
            logger.debug("DataPipeline: outcome persist failed", exc_info=True)

    async def get_training_stats(self) -> dict:
        """Get stats on collected training data."""
        stats = {
            "enriched_signals": await self._redis.llen("oracle:enriched_signals"),
            "signal_embeddings": await self._redis.llen("oracle:signal_embeddings"),
            "journal_entries": await self._redis.llen("oracle:trade_journal"),
            "resolved_outcomes": await self._redis.llen("oracle:journal_resolved"),
        }

        try:
            from oracle_shared.db import get_session
            from oracle_shared.db.models import SignalRow, TradeExecutionRow
            from sqlalchemy import select, func
            async with get_session() as session:
                stats["postgres_signals"] = (await session.execute(
                    select(func.count()).select_from(SignalRow)
                )).scalar_one()
                stats["postgres_executions"] = (await session.execute(
                    select(func.count()).select_from(TradeExecutionRow)
                )).scalar_one()
        except Exception:
            stats["postgres_signals"] = "unavailable"
            stats["postgres_executions"] = "unavailable"

        return stats
