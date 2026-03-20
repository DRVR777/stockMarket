"""AnomalyEvent contract — emitted by whale-detector (WADE) when an on-chain fill exceeds anomaly thresholds.

Published to ``oracle:anomaly_event``. Consumed by reasoning-engine and operator-dashboard.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import ClassVar, Optional
from datetime import datetime
import uuid


class AnomalyEvent(BaseModel):
    event_id:            str      = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:           datetime
    wallet_address:      str
    market_id:           str
    outcome:             str                       # "YES" | "NO" | specific option
    notional_usd:        float
    anomaly_score:       float                     # 0.0–1.0
    wallet_profile:      Optional[dict] = None     # WalletProfile.model_dump() if in registry
    trigger_reasons:     list[str]                 # e.g. ["large_order", "tier_1_wallet"]
    copy_trade_eligible: bool
    cascade_wallets:     Optional[list[str]] = None
    source_signal_id:    str                       # the polygon_clob Signal that triggered this

    # Redis channel
    CHANNEL: ClassVar[str] = "oracle:anomaly_event"
