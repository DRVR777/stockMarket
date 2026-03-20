"""Signal contract — canonical schema for all data ingested by signal-ingestion (SIL).

Every adapter normalizes raw source data into a Signal and publishes it to the
``oracle:signal`` Redis channel. Downstream consumers (whale-detector, osint-fusion)
validate incoming messages against this schema.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, ClassVar, Optional
from enum import Enum
from datetime import datetime
import uuid


class SignalCategory(str, Enum):
    ON_CHAIN    = "on_chain"
    NEWS        = "news"
    SOCIAL      = "social"
    PRICE       = "price"
    AI          = "ai_generated"


class SourceId(str, Enum):
    POLYMARKET_REST = "polymarket_rest"
    POLYMARKET_WS   = "polymarket_ws"
    POLYGON_CLOB    = "polygon_clob"
    NEWSAPI         = "newsapi"
    WIKIPEDIA       = "wikipedia"
    REDDIT          = "reddit"
    BIRDEYE         = "birdeye"
    AI_OPINION      = "ai_opinion"


class Signal(BaseModel):
    signal_id:   str           = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id:   SourceId
    timestamp:   datetime
    category:    SignalCategory
    raw_payload: dict[str, Any]
    confidence:  Optional[float] = None   # 0.0–1.0 where applicable
    market_ids:  Optional[list[str]] = None  # pre-associated market IDs if SIL can determine

    # Redis channel
    CHANNEL: ClassVar[str] = "oracle:signal"
