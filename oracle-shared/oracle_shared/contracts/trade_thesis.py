"""TradeThesis contract — the multi-pass reasoning output for a single market position decision.

Published to ``oracle:trade_thesis``. Produced by reasoning-engine (RE). Consumed by
solana-executor, operator-dashboard, and knowledge-base.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import ClassVar, Optional
from enum import Enum
from datetime import datetime
import uuid


class ThesisDecision(str, Enum):
    EXECUTE         = "execute"
    FLAG_FOR_REVIEW = "flag_for_review"
    SKIP            = "skip"


class ThesisOutcome(str, Enum):
    WIN  = "win"
    LOSS = "loss"
    VOID = "void"


class Hypothesis(BaseModel):
    side:     str        # "YES" or "NO"
    argument: str
    evidence: list[str]


class EvidenceWeight(BaseModel):
    hypothesis_side: str
    score:           float  # 0.0–1.0
    reasoning:       str


class HistoricalAnalogue(BaseModel):
    thesis_id:  str
    similarity: float
    outcome:    Optional[str] = None  # ThesisOutcome value if resolved


class ContextAssembly(BaseModel):
    market_state:         dict          # MarketState.model_dump()
    anomaly_events:       list[dict]    # AnomalyEvent.model_dump() list
    historical_analogues: list[HistoricalAnalogue]
    assembled_at:         datetime


class TradeThesis(BaseModel):
    thesis_id:                  str            = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at:                 datetime
    market_id:                  str
    market_question:            str
    direction:                  str            # "YES" | "NO"
    re_probability_estimate:    float          # 0.0–1.0
    market_implied_probability: float
    probability_delta:          float          # re_prob - market_prob
    confidence_score:           float          # 0.0–1.0
    decision:                   ThesisDecision
    recommended_position_usd:   Optional[float] = None
    hypotheses:                 list[Hypothesis]
    evidence_weights:           list[EvidenceWeight]
    context_assembly:           ContextAssembly
    outcome:                    Optional[ThesisOutcome] = None
    outcome_label_at:           Optional[datetime] = None
    vault_path:                 Optional[str] = None

    # Redis channel
    CHANNEL: ClassVar[str] = "oracle:trade_thesis"
