"""MarketState contract — rolling state snapshot for a single Polymarket market.

Maintained in Redis by osint-fusion. Published to ``oracle:market_state`` on updates.
Read by reasoning-engine for context assembly.
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import ClassVar, Optional
from datetime import datetime

RECENT_INSIGHTS_WINDOW = 20  # max Insight objects kept in rolling window


class MarketState(BaseModel):
    market_id:              str
    market_question:        str
    current_price_yes:      float          # 0.0–1.0
    resolution_deadline:    datetime
    liquidity_usd:          float
    last_price_updated:     datetime
    last_signal_at:         Optional[datetime] = None
    recent_insights:        list[dict] = []  # last N Insight.model_dump(), newest first
    semantic_state_summary: str = ""
    signal_count_24h:       int = 0
    whale_event_count_24h:  int = 0

    # Redis keys
    CHANNEL: ClassVar[str] = "oracle:market_state"
    STATE_KEY_PREFIX: ClassVar[str] = "oracle:state:markets"  # HSET field = market_id
