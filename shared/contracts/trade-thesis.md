# Contract: TradeThesis

## Status
defined — build against this shape

## Produced By
reasoning-engine

## Consumed By
knowledge-base, solana-executor, operator-dashboard

## Shape

```python
from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum
import uuid
from datetime import datetime

class ThesisDecision(str, Enum):
    EXECUTE        = "execute"
    FLAG_FOR_REVIEW = "flag_for_review"
    SKIP           = "skip"

class ThesisOutcome(str, Enum):
    WIN  = "win"
    LOSS = "loss"
    VOID = "void"   # market resolved void / cancelled

class Hypothesis(BaseModel):
    side:      str        # "YES" or "NO"
    argument:  str        # RE's argument for this side
    evidence:  list[str]  # supporting evidence items cited

class EvidenceWeight(BaseModel):
    hypothesis_side: str
    score:           float   # 0.0–1.0
    reasoning:       str

class HistoricalAnalogue(BaseModel):
    thesis_id:    str
    similarity:   float  # ChromaDB cosine similarity score
    outcome:      Optional[str] = None  # ThesisOutcome value if resolved

class ContextAssembly(BaseModel):
    market_state:         dict          # MarketState.model_dump() at time of analysis
    anomaly_events:       list[dict]    # AnomalyEvent.model_dump() list for this market
    historical_analogues: list[HistoricalAnalogue]
    assembled_at:         datetime

class TradeThesis(BaseModel):
    thesis_id:                  str           = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at:                 datetime
    market_id:                  str
    market_question:            str
    direction:                  str           # "YES" | "NO"
    re_probability_estimate:    float         # 0.0–1.0
    market_implied_probability: float         # current market price at analysis time
    probability_delta:          float         # re_probability_estimate - market_implied_probability
    confidence_score:           float         # 0.0–1.0
    decision:                   ThesisDecision
    recommended_position_usd:   Optional[float] = None  # null if decision != execute
    hypotheses:                 list[Hypothesis]
    evidence_weights:           list[EvidenceWeight]
    context_assembly:           ContextAssembly
    # Populated after resolution by KBPM:
    outcome:                    Optional[ThesisOutcome] = None
    outcome_label_at:           Optional[datetime] = None
    vault_path:                 Optional[str] = None  # path in KBPM /theses/ vault
```

## recommended_position_usd calculation
RE computes this when decision == execute:
`min(max_position_usd_param, confidence_score * probability_delta * base_stake_usd_param)`
where params are loaded from `oracle:state:params`.

## Redis channel
`oracle:trade_thesis`

## Serialization
`thesis.model_dump_json()` → publish. `TradeThesis.model_validate_json(msg)` → consume.
