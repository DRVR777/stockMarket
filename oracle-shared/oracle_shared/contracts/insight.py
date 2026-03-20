"""Insight contract — produced by osint-fusion (OSFE) after embedding a Signal and matching it to active markets.

Published to ``oracle:insight``. Consumed by reasoning-engine to assemble market context.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import ClassVar, Optional
from datetime import datetime
import uuid

# Default credibility weights by Signal.category value
DEFAULT_CREDIBILITY_WEIGHTS: dict[str, float] = {
    "on_chain":    1.5,
    "news":        1.0,
    "social":      0.6,
    "price":       1.2,
    "ai_generated": 0.8,
}

# Minimum similarity score to include a market in associated_market_ids
SIMILARITY_THRESHOLD = 0.65


class Insight(BaseModel):
    insight_id:                str            = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:                 datetime
    source_signal_id:          str
    source_category:           str            # mirrors Signal.category value
    associated_market_ids:     list[str]      # similarity >= SIMILARITY_THRESHOLD
    similarity_scores:         dict[str, float]  # {market_id: score 0.0–1.0}
    semantic_summary:          str
    source_credibility_weight: float          # 0.0–2.0
    raw_text:                  str            # the text that was embedded

    # Redis channel
    CHANNEL: ClassVar[str] = "oracle:insight"
