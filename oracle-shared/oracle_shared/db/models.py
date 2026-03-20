"""SQLAlchemy ORM models for ORACLE persistent storage.

Each table mirrors a Pydantic contract from oracle_shared.contracts.
Pydantic models remain the canonical in-flight schema; these tables
are the durable store.

Conversion helpers:
    pydantic_obj → row:  MyTable.from_contract(pydantic_obj)
    row → pydantic_obj:  row.to_contract()
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORACLE tables."""
    pass


# ── Signals ───────────────────────────────────────────────────────────────────

class SignalRow(Base):
    __tablename__ = "signals"

    signal_id:   Mapped[str]      = mapped_column(String(36), primary_key=True)
    source_id:   Mapped[str]      = mapped_column(String(32), nullable=False)
    timestamp:   Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    category:    Mapped[str]      = mapped_column(String(20), nullable=False)
    raw_payload: Mapped[dict]     = mapped_column(JSONB, nullable=False)
    confidence:  Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_ids:  Mapped[Optional[list]]  = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_signals_timestamp", "timestamp"),
        Index("ix_signals_source_category", "source_id", "category"),
    )

    @classmethod
    def from_contract(cls, sig: Any) -> SignalRow:
        from oracle_shared.contracts.signal import Signal
        s: Signal = sig
        return cls(
            signal_id=s.signal_id,
            source_id=s.source_id.value,
            timestamp=s.timestamp,
            category=s.category.value,
            raw_payload=s.raw_payload,
            confidence=s.confidence,
            market_ids=s.market_ids,
        )

    def to_contract(self) -> Any:
        from oracle_shared.contracts.signal import Signal
        return Signal.model_validate({
            "signal_id": self.signal_id,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "category": self.category,
            "raw_payload": self.raw_payload,
            "confidence": self.confidence,
            "market_ids": self.market_ids,
        })


# ── Anomaly Events ────────────────────────────────────────────────────────────

class AnomalyEventRow(Base):
    __tablename__ = "anomaly_events"

    event_id:            Mapped[str]      = mapped_column(String(36), primary_key=True)
    timestamp:           Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    wallet_address:      Mapped[str]      = mapped_column(String(42), nullable=False)
    market_id:           Mapped[str]      = mapped_column(String(128), nullable=False)
    outcome:             Mapped[str]      = mapped_column(String(64), nullable=False)
    notional_usd:        Mapped[float]    = mapped_column(Float, nullable=False)
    anomaly_score:       Mapped[float]    = mapped_column(Float, nullable=False)
    wallet_profile:      Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    trigger_reasons:     Mapped[list]     = mapped_column(JSONB, nullable=False)
    copy_trade_eligible: Mapped[bool]     = mapped_column(Boolean, nullable=False)
    cascade_wallets:     Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    source_signal_id:    Mapped[str]      = mapped_column(String(36), nullable=False)

    __table_args__ = (
        Index("ix_anomaly_events_market", "market_id", "timestamp"),
        Index("ix_anomaly_events_wallet", "wallet_address"),
    )

    @classmethod
    def from_contract(cls, ae: Any) -> AnomalyEventRow:
        from oracle_shared.contracts.anomaly_event import AnomalyEvent
        e: AnomalyEvent = ae
        return cls(
            event_id=e.event_id,
            timestamp=e.timestamp,
            wallet_address=e.wallet_address,
            market_id=e.market_id,
            outcome=e.outcome,
            notional_usd=e.notional_usd,
            anomaly_score=e.anomaly_score,
            wallet_profile=e.wallet_profile,
            trigger_reasons=e.trigger_reasons,
            copy_trade_eligible=e.copy_trade_eligible,
            cascade_wallets=e.cascade_wallets,
            source_signal_id=e.source_signal_id,
        )

    def to_contract(self) -> Any:
        from oracle_shared.contracts.anomaly_event import AnomalyEvent
        return AnomalyEvent.model_validate({
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "wallet_address": self.wallet_address,
            "market_id": self.market_id,
            "outcome": self.outcome,
            "notional_usd": self.notional_usd,
            "anomaly_score": self.anomaly_score,
            "wallet_profile": self.wallet_profile,
            "trigger_reasons": self.trigger_reasons,
            "copy_trade_eligible": self.copy_trade_eligible,
            "cascade_wallets": self.cascade_wallets,
            "source_signal_id": self.source_signal_id,
        })


# ── Insights ──────────────────────────────────────────────────────────────────

class InsightRow(Base):
    __tablename__ = "insights"

    insight_id:                Mapped[str]      = mapped_column(String(36), primary_key=True)
    timestamp:                 Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_signal_id:          Mapped[str]      = mapped_column(String(36), nullable=False)
    source_category:           Mapped[str]      = mapped_column(String(20), nullable=False)
    associated_market_ids:     Mapped[list]     = mapped_column(JSONB, nullable=False)
    similarity_scores:         Mapped[dict]     = mapped_column(JSONB, nullable=False)
    semantic_summary:          Mapped[str]      = mapped_column(Text, nullable=False)
    source_credibility_weight: Mapped[float]    = mapped_column(Float, nullable=False)
    raw_text:                  Mapped[str]      = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("ix_insights_timestamp", "timestamp"),
    )

    @classmethod
    def from_contract(cls, ins: Any) -> InsightRow:
        from oracle_shared.contracts.insight import Insight
        i: Insight = ins
        return cls(
            insight_id=i.insight_id,
            timestamp=i.timestamp,
            source_signal_id=i.source_signal_id,
            source_category=i.source_category,
            associated_market_ids=i.associated_market_ids,
            similarity_scores=i.similarity_scores,
            semantic_summary=i.semantic_summary,
            source_credibility_weight=i.source_credibility_weight,
            raw_text=i.raw_text,
        )

    def to_contract(self) -> Any:
        from oracle_shared.contracts.insight import Insight
        return Insight.model_validate({
            "insight_id": self.insight_id,
            "timestamp": self.timestamp,
            "source_signal_id": self.source_signal_id,
            "source_category": self.source_category,
            "associated_market_ids": self.associated_market_ids,
            "similarity_scores": self.similarity_scores,
            "semantic_summary": self.semantic_summary,
            "source_credibility_weight": self.source_credibility_weight,
            "raw_text": self.raw_text,
        })


# ── Trade Theses ──────────────────────────────────────────────────────────────

class TradeThesisRow(Base):
    __tablename__ = "trade_theses"

    thesis_id:                  Mapped[str]      = mapped_column(String(36), primary_key=True)
    created_at:                 Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    market_id:                  Mapped[str]      = mapped_column(String(128), nullable=False)
    market_question:            Mapped[str]      = mapped_column(Text, nullable=False)
    direction:                  Mapped[str]      = mapped_column(String(10), nullable=False)
    re_probability_estimate:    Mapped[float]    = mapped_column(Float, nullable=False)
    market_implied_probability: Mapped[float]    = mapped_column(Float, nullable=False)
    probability_delta:          Mapped[float]    = mapped_column(Float, nullable=False)
    confidence_score:           Mapped[float]    = mapped_column(Float, nullable=False)
    decision:                   Mapped[str]      = mapped_column(String(20), nullable=False)
    recommended_position_usd:   Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hypotheses:                 Mapped[list]     = mapped_column(JSONB, nullable=False)
    evidence_weights:           Mapped[list]     = mapped_column(JSONB, nullable=False)
    context_assembly:           Mapped[dict]     = mapped_column(JSONB, nullable=False)
    outcome:                    Mapped[Optional[str]]      = mapped_column(String(10), nullable=True)
    outcome_label_at:           Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    vault_path:                 Mapped[Optional[str]]      = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_theses_market", "market_id", "created_at"),
        Index("ix_theses_decision", "decision"),
        Index("ix_theses_created", "created_at"),
    )

    @classmethod
    def from_contract(cls, thesis: Any) -> TradeThesisRow:
        from oracle_shared.contracts.trade_thesis import TradeThesis
        t: TradeThesis = thesis
        return cls(
            thesis_id=t.thesis_id,
            created_at=t.created_at,
            market_id=t.market_id,
            market_question=t.market_question,
            direction=t.direction,
            re_probability_estimate=t.re_probability_estimate,
            market_implied_probability=t.market_implied_probability,
            probability_delta=t.probability_delta,
            confidence_score=t.confidence_score,
            decision=t.decision.value,
            recommended_position_usd=t.recommended_position_usd,
            hypotheses=[h.model_dump() for h in t.hypotheses],
            evidence_weights=[e.model_dump() for e in t.evidence_weights],
            context_assembly=t.context_assembly.model_dump(),
            outcome=t.outcome.value if t.outcome else None,
            outcome_label_at=t.outcome_label_at,
            vault_path=t.vault_path,
        )

    def to_contract(self) -> Any:
        from oracle_shared.contracts.trade_thesis import TradeThesis
        return TradeThesis.model_validate({
            "thesis_id": self.thesis_id,
            "created_at": self.created_at,
            "market_id": self.market_id,
            "market_question": self.market_question,
            "direction": self.direction,
            "re_probability_estimate": self.re_probability_estimate,
            "market_implied_probability": self.market_implied_probability,
            "probability_delta": self.probability_delta,
            "confidence_score": self.confidence_score,
            "decision": self.decision,
            "recommended_position_usd": self.recommended_position_usd,
            "hypotheses": self.hypotheses,
            "evidence_weights": self.evidence_weights,
            "context_assembly": self.context_assembly,
            "outcome": self.outcome,
            "outcome_label_at": self.outcome_label_at,
            "vault_path": self.vault_path,
        })


# ── Wallet Profiles ───────────────────────────────────────────────────────────

class WalletProfileRow(Base):
    __tablename__ = "wallet_profiles"

    wallet_address:             Mapped[str]      = mapped_column(String(42), primary_key=True)
    reputation_tier:            Mapped[str]      = mapped_column(String(16), nullable=False, default="Unknown")
    tier_assignment_method:     Mapped[str]      = mapped_column(String(16), nullable=False, default="algorithmic")
    historical_pnl_usd:        Mapped[float]    = mapped_column(Float, nullable=False, default=0.0)
    win_rate:                   Mapped[float]    = mapped_column(Float, nullable=False, default=0.0)
    typical_position_size_usd:  Mapped[float]    = mapped_column(Float, nullable=False, default=0.0)
    market_category_preference: Mapped[list]     = mapped_column(JSONB, nullable=False, default=list)
    first_seen_at:              Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_active_at:             Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    total_trades_tracked:       Mapped[int]      = mapped_column(Integer, nullable=False, default=0)
    notes:                      Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_wallets_tier", "reputation_tier"),
        Index("ix_wallets_active", "last_active_at"),
    )

    @classmethod
    def from_contract(cls, wp: Any) -> WalletProfileRow:
        from oracle_shared.contracts.wallet_profile import WalletProfile
        w: WalletProfile = wp
        return cls(
            wallet_address=w.wallet_address,
            reputation_tier=w.reputation_tier.value,
            tier_assignment_method=w.tier_assignment_method,
            historical_pnl_usd=w.historical_pnl_usd,
            win_rate=w.win_rate,
            typical_position_size_usd=w.typical_position_size_usd,
            market_category_preference=w.market_category_preference,
            first_seen_at=w.first_seen_at,
            last_active_at=w.last_active_at,
            total_trades_tracked=w.total_trades_tracked,
            notes=w.notes,
        )

    def to_contract(self) -> Any:
        from oracle_shared.contracts.wallet_profile import WalletProfile
        return WalletProfile.model_validate({
            "wallet_address": self.wallet_address,
            "reputation_tier": self.reputation_tier,
            "tier_assignment_method": self.tier_assignment_method,
            "historical_pnl_usd": self.historical_pnl_usd,
            "win_rate": self.win_rate,
            "typical_position_size_usd": self.typical_position_size_usd,
            "market_category_preference": self.market_category_preference,
            "first_seen_at": self.first_seen_at,
            "last_active_at": self.last_active_at,
            "total_trades_tracked": self.total_trades_tracked,
            "notes": self.notes,
        })


# ── Trade Executions ──────────────────────────────────────────────────────────

class TradeExecutionRow(Base):
    __tablename__ = "trade_executions"

    execution_id:             Mapped[str]      = mapped_column(String(36), primary_key=True)
    thesis_id:                Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    market_id:                Mapped[str]      = mapped_column(String(128), nullable=False)
    market_type:              Mapped[str]      = mapped_column(String(16), nullable=False)
    direction:                Mapped[str]      = mapped_column(String(10), nullable=False)
    outcome:                  Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    entry_price:              Mapped[float]    = mapped_column(Float, nullable=False)
    size_usd:                 Mapped[float]    = mapped_column(Float, nullable=False)
    executed_at:              Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    execution_source:         Mapped[str]      = mapped_column(String(24), nullable=False)
    status:                   Mapped[str]      = mapped_column(String(10), nullable=False, default="open")
    exit_price:               Mapped[Optional[float]]    = mapped_column(Float, nullable=True)
    exit_at:                  Mapped[Optional[datetime]]  = mapped_column(DateTime(timezone=True), nullable=True)
    exit_reason:              Mapped[Optional[str]]       = mapped_column(String(20), nullable=True)
    realized_pnl_usd:        Mapped[Optional[float]]     = mapped_column(Float, nullable=True)
    circuit_breaker_checked:  Mapped[bool]     = mapped_column(Boolean, nullable=False, default=True)
    tx_hash:                  Mapped[Optional[str]]  = mapped_column(String(128), nullable=True)
    copy_trade_source_wallet: Mapped[Optional[str]]  = mapped_column(String(42), nullable=True)

    __table_args__ = (
        Index("ix_executions_market", "market_id"),
        Index("ix_executions_thesis", "thesis_id"),
        Index("ix_executions_status", "status"),
        Index("ix_executions_time", "executed_at"),
    )

    @classmethod
    def from_contract(cls, te: Any) -> TradeExecutionRow:
        from oracle_shared.contracts.trade_execution import TradeExecution
        t: TradeExecution = te
        return cls(
            execution_id=t.execution_id,
            thesis_id=t.thesis_id,
            market_id=t.market_id,
            market_type=t.market_type.value,
            direction=t.direction,
            outcome=t.outcome,
            entry_price=t.entry_price,
            size_usd=t.size_usd,
            executed_at=t.executed_at,
            execution_source=t.execution_source.value,
            status=t.status.value,
            exit_price=t.exit_price,
            exit_at=t.exit_at,
            exit_reason=t.exit_reason.value if t.exit_reason else None,
            realized_pnl_usd=t.realized_pnl_usd,
            circuit_breaker_checked=t.circuit_breaker_checked,
            tx_hash=t.tx_hash,
            copy_trade_source_wallet=t.copy_trade_source_wallet,
        )

    def to_contract(self) -> Any:
        from oracle_shared.contracts.trade_execution import TradeExecution
        return TradeExecution.model_validate({
            "execution_id": self.execution_id,
            "thesis_id": self.thesis_id,
            "market_id": self.market_id,
            "market_type": self.market_type,
            "direction": self.direction,
            "outcome": self.outcome,
            "entry_price": self.entry_price,
            "size_usd": self.size_usd,
            "executed_at": self.executed_at,
            "execution_source": self.execution_source,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_at": self.exit_at,
            "exit_reason": self.exit_reason,
            "realized_pnl_usd": self.realized_pnl_usd,
            "circuit_breaker_checked": self.circuit_breaker_checked,
            "tx_hash": self.tx_hash,
            "copy_trade_source_wallet": self.copy_trade_source_wallet,
        })


# ── Post-Mortems ──────────────────────────────────────────────────────────────

class PostMortemRow(Base):
    __tablename__ = "post_mortems"

    postmortem_id:                   Mapped[str]      = mapped_column(String(36), primary_key=True)
    generated_at:                    Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    market_id:                       Mapped[str]      = mapped_column(String(128), nullable=False)
    market_question:                 Mapped[str]      = mapped_column(Text, nullable=False)
    thesis_id:                       Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    execution_id:                    Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    market_resolved_as:              Mapped[str]      = mapped_column(String(32), nullable=False)
    thesis_was_correct:              Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    realized_pnl_usd:               Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    signals_present:                 Mapped[list]     = mapped_column(JSONB, nullable=False)
    what_the_thesis_said:            Mapped[str]      = mapped_column(Text, nullable=False)
    what_happened:                   Mapped[str]      = mapped_column(Text, nullable=False)
    what_would_have_changed_outcome: Mapped[str]      = mapped_column(Text, nullable=False)
    source_weight_updates:           Mapped[dict]     = mapped_column(JSONB, nullable=False)
    vault_path:                      Mapped[str]      = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("ix_postmortems_market", "market_id"),
        Index("ix_postmortems_thesis", "thesis_id"),
        Index("ix_postmortems_time", "generated_at"),
    )

    @classmethod
    def from_contract(cls, pm: Any) -> PostMortemRow:
        from oracle_shared.contracts.post_mortem import PostMortem
        p: PostMortem = pm
        return cls(
            postmortem_id=p.postmortem_id,
            generated_at=p.generated_at,
            market_id=p.market_id,
            market_question=p.market_question,
            thesis_id=p.thesis_id,
            execution_id=p.execution_id,
            market_resolved_as=p.market_resolved_as,
            thesis_was_correct=p.thesis_was_correct,
            realized_pnl_usd=p.realized_pnl_usd,
            signals_present=[s.model_dump() for s in p.signals_present],
            what_the_thesis_said=p.what_the_thesis_said,
            what_happened=p.what_happened,
            what_would_have_changed_outcome=p.what_would_have_changed_outcome,
            source_weight_updates=p.source_weight_updates,
            vault_path=p.vault_path,
        )

    def to_contract(self) -> Any:
        from oracle_shared.contracts.post_mortem import PostMortem
        return PostMortem.model_validate({
            "postmortem_id": self.postmortem_id,
            "generated_at": self.generated_at,
            "market_id": self.market_id,
            "market_question": self.market_question,
            "thesis_id": self.thesis_id,
            "execution_id": self.execution_id,
            "market_resolved_as": self.market_resolved_as,
            "thesis_was_correct": self.thesis_was_correct,
            "realized_pnl_usd": self.realized_pnl_usd,
            "signals_present": self.signals_present,
            "what_the_thesis_said": self.what_the_thesis_said,
            "what_happened": self.what_happened,
            "what_would_have_changed_outcome": self.what_would_have_changed_outcome,
            "source_weight_updates": self.source_weight_updates,
            "vault_path": self.vault_path,
        })


# ── Operator Alerts ───────────────────────────────────────────────────────────

class OperatorAlertRow(Base):
    __tablename__ = "operator_alerts"

    alert_id:        Mapped[str]      = mapped_column(String(36), primary_key=True)
    created_at:      Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    alert_type:      Mapped[str]      = mapped_column(String(20), nullable=False)
    severity:        Mapped[str]      = mapped_column(String(20), nullable=False)
    title:           Mapped[str]      = mapped_column(String(80), nullable=False)
    body:            Mapped[str]      = mapped_column(Text, nullable=False)
    action_required: Mapped[bool]     = mapped_column(Boolean, nullable=False)
    action_options:  Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    linked_event_id: Mapped[Optional[str]]  = mapped_column(String(36), nullable=True)
    acknowledged:    Mapped[bool]     = mapped_column(Boolean, nullable=False, default=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    action_taken:    Mapped[Optional[str]]      = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_alerts_type", "alert_type", "created_at"),
        Index("ix_alerts_unacked", "acknowledged", "created_at"),
    )

    @classmethod
    def from_contract(cls, alert: Any) -> OperatorAlertRow:
        from oracle_shared.contracts.operator_alert import OperatorAlert
        a: OperatorAlert = alert
        return cls(
            alert_id=a.alert_id,
            created_at=a.created_at,
            alert_type=a.alert_type.value,
            severity=a.severity.value,
            title=a.title,
            body=a.body,
            action_required=a.action_required,
            action_options=a.action_options,
            linked_event_id=a.linked_event_id,
            acknowledged=a.acknowledged,
            acknowledged_at=a.acknowledged_at,
            action_taken=a.action_taken,
        )

    def to_contract(self) -> Any:
        from oracle_shared.contracts.operator_alert import OperatorAlert
        return OperatorAlert.model_validate({
            "alert_id": self.alert_id,
            "created_at": self.created_at,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "body": self.body,
            "action_required": self.action_required,
            "action_options": self.action_options,
            "linked_event_id": self.linked_event_id,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
            "action_taken": self.action_taken,
        })
