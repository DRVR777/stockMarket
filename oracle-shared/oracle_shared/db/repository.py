"""Async repository layer for ORACLE persistent storage.

Provides typed CRUD operations for each entity. All methods accept
an AsyncSession — callers use ``get_session()`` from ``oracle_shared.db``.

Usage::

    from oracle_shared.db import get_session
    from oracle_shared.db.repository import ThesisRepo

    async with get_session() as session:
        await ThesisRepo.save(session, trade_thesis_pydantic)
        theses = await ThesisRepo.by_market(session, "market_123", limit=10)
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Sequence

from sqlalchemy import cast, select, update, desc
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from oracle_shared.db.models import (
    AnomalyEventRow,
    InsightRow,
    OperatorAlertRow,
    PostMortemRow,
    SignalRow,
    TradeExecutionRow,
    TradeThesisRow,
    WalletProfileRow,
)


# ── Signals ───────────────────────────────────────────────────────────────────

class SignalRepo:

    @staticmethod
    async def save(session: AsyncSession, signal: Any) -> None:
        row = SignalRow.from_contract(signal)
        session.add(row)

    @staticmethod
    async def save_batch(session: AsyncSession, signals: list[Any]) -> None:
        session.add_all([SignalRow.from_contract(s) for s in signals])

    @staticmethod
    async def recent(
        session: AsyncSession,
        limit: int = 50,
        source_id: str | None = None,
    ) -> Sequence[SignalRow]:
        stmt = select(SignalRow).order_by(desc(SignalRow.timestamp)).limit(limit)
        if source_id:
            stmt = stmt.where(SignalRow.source_id == source_id)
        result = await session.execute(stmt)
        return result.scalars().all()


# ── Anomaly Events ────────────────────────────────────────────────────────────

class AnomalyEventRepo:

    @staticmethod
    async def save(session: AsyncSession, event: Any) -> None:
        row = AnomalyEventRow.from_contract(event)
        session.add(row)

    @staticmethod
    async def by_market(
        session: AsyncSession,
        market_id: str,
        limit: int = 10,
    ) -> Sequence[AnomalyEventRow]:
        stmt = (
            select(AnomalyEventRow)
            .where(AnomalyEventRow.market_id == market_id)
            .order_by(desc(AnomalyEventRow.timestamp))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def by_wallet(
        session: AsyncSession,
        wallet_address: str,
        limit: int = 20,
    ) -> Sequence[AnomalyEventRow]:
        stmt = (
            select(AnomalyEventRow)
            .where(AnomalyEventRow.wallet_address == wallet_address)
            .order_by(desc(AnomalyEventRow.timestamp))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()


# ── Insights ──────────────────────────────────────────────────────────────────

class InsightRepo:

    @staticmethod
    async def save(session: AsyncSession, insight: Any) -> None:
        row = InsightRow.from_contract(insight)
        session.add(row)

    @staticmethod
    async def recent_for_market(
        session: AsyncSession,
        market_id: str,
        limit: int = 20,
    ) -> Sequence[InsightRow]:
        """Insights whose associated_market_ids contain market_id."""
        # Use @> with explicit JSONB cast to check array containment
        stmt = (
            select(InsightRow)
            .where(
                InsightRow.associated_market_ids.op("@>")(
                    cast([market_id], JSONB)
                )
            )
            .order_by(desc(InsightRow.timestamp))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()


# ── Trade Theses ──────────────────────────────────────────────────────────────

class ThesisRepo:

    @staticmethod
    async def save(session: AsyncSession, thesis: Any) -> None:
        row = TradeThesisRow.from_contract(thesis)
        session.add(row)

    @staticmethod
    async def get(session: AsyncSession, thesis_id: str) -> Optional[TradeThesisRow]:
        stmt = select(TradeThesisRow).where(TradeThesisRow.thesis_id == thesis_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def by_market(
        session: AsyncSession,
        market_id: str,
        limit: int = 10,
    ) -> Sequence[TradeThesisRow]:
        stmt = (
            select(TradeThesisRow)
            .where(TradeThesisRow.market_id == market_id)
            .order_by(desc(TradeThesisRow.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def by_decision(
        session: AsyncSession,
        decision: str,
        limit: int = 20,
    ) -> Sequence[TradeThesisRow]:
        stmt = (
            select(TradeThesisRow)
            .where(TradeThesisRow.decision == decision)
            .order_by(desc(TradeThesisRow.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def set_outcome(
        session: AsyncSession,
        thesis_id: str,
        outcome: str,
        label_at: datetime | None = None,
    ) -> None:
        stmt = (
            update(TradeThesisRow)
            .where(TradeThesisRow.thesis_id == thesis_id)
            .values(
                outcome=outcome,
                outcome_label_at=label_at or datetime.now(timezone.utc),
            )
        )
        await session.execute(stmt)

    @staticmethod
    async def unresolved(
        session: AsyncSession,
        limit: int = 50,
    ) -> Sequence[TradeThesisRow]:
        """Theses with decision=execute that haven't been resolved yet."""
        from oracle_shared.contracts.trade_thesis import ThesisDecision
        stmt = (
            select(TradeThesisRow)
            .where(
                TradeThesisRow.decision == ThesisDecision.EXECUTE.value,
                TradeThesisRow.outcome.is_(None),
            )
            .order_by(desc(TradeThesisRow.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()


# ── Wallet Profiles ───────────────────────────────────────────────────────────

class WalletRepo:

    @staticmethod
    async def upsert(session: AsyncSession, profile: Any) -> None:
        row = WalletProfileRow.from_contract(profile)
        # merge() is sync in SQLAlchemy ORM — do not await
        session.merge(row)

    @staticmethod
    async def get(session: AsyncSession, wallet_address: str) -> Optional[WalletProfileRow]:
        stmt = select(WalletProfileRow).where(
            WalletProfileRow.wallet_address == wallet_address
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def by_tier(
        session: AsyncSession,
        tier: str,
        limit: int = 50,
    ) -> Sequence[WalletProfileRow]:
        stmt = (
            select(WalletProfileRow)
            .where(WalletProfileRow.reputation_tier == tier)
            .order_by(desc(WalletProfileRow.last_active_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()


# ── Trade Executions ──────────────────────────────────────────────────────────

class ExecutionRepo:

    @staticmethod
    async def save(session: AsyncSession, execution: Any) -> None:
        row = TradeExecutionRow.from_contract(execution)
        session.add(row)

    @staticmethod
    async def open_positions(session: AsyncSession) -> Sequence[TradeExecutionRow]:
        from oracle_shared.contracts.trade_execution import ExecutionStatus
        stmt = (
            select(TradeExecutionRow)
            .where(TradeExecutionRow.status == ExecutionStatus.OPEN.value)
            .order_by(desc(TradeExecutionRow.executed_at))
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def close_position(
        session: AsyncSession,
        execution_id: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl: float,
    ) -> None:
        from oracle_shared.contracts.trade_execution import ExecutionStatus
        stmt = (
            update(TradeExecutionRow)
            .where(TradeExecutionRow.execution_id == execution_id)
            .values(
                status=ExecutionStatus.CLOSED.value,
                exit_price=exit_price,
                exit_at=datetime.now(timezone.utc),
                exit_reason=exit_reason,
                realized_pnl_usd=realized_pnl,
            )
        )
        await session.execute(stmt)

    @staticmethod
    async def pnl_since(
        session: AsyncSession,
        since: datetime,
    ) -> float:
        """Sum realized PnL for closed positions since a given time."""
        from sqlalchemy import func
        from oracle_shared.contracts.trade_execution import ExecutionStatus
        stmt = (
            select(func.coalesce(func.sum(TradeExecutionRow.realized_pnl_usd), 0.0))
            .where(
                TradeExecutionRow.status == ExecutionStatus.CLOSED.value,
                TradeExecutionRow.exit_at >= since,
            )
        )
        result = await session.execute(stmt)
        return float(result.scalar_one())


# ── Post-Mortems ──────────────────────────────────────────────────────────────

class PostMortemRepo:

    @staticmethod
    async def save(session: AsyncSession, pm: Any) -> None:
        row = PostMortemRow.from_contract(pm)
        session.add(row)

    @staticmethod
    async def by_market(
        session: AsyncSession,
        market_id: str,
    ) -> Sequence[PostMortemRow]:
        stmt = (
            select(PostMortemRow)
            .where(PostMortemRow.market_id == market_id)
            .order_by(desc(PostMortemRow.generated_at))
        )
        result = await session.execute(stmt)
        return result.scalars().all()


# ── Operator Alerts ───────────────────────────────────────────────────────────

class AlertRepo:

    @staticmethod
    async def save(session: AsyncSession, alert: Any) -> None:
        row = OperatorAlertRow.from_contract(alert)
        session.add(row)

    @staticmethod
    async def unacknowledged(
        session: AsyncSession,
        limit: int = 20,
    ) -> Sequence[OperatorAlertRow]:
        stmt = (
            select(OperatorAlertRow)
            .where(OperatorAlertRow.acknowledged.is_(False))
            .order_by(desc(OperatorAlertRow.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def acknowledge(
        session: AsyncSession,
        alert_id: str,
        action_taken: str | None = None,
    ) -> None:
        stmt = (
            update(OperatorAlertRow)
            .where(OperatorAlertRow.alert_id == alert_id)
            .values(
                acknowledged=True,
                acknowledged_at=datetime.now(timezone.utc),
                action_taken=action_taken,
            )
        )
        await session.execute(stmt)
