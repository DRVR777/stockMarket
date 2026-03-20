"""WalletProfile contract — reputation and historical statistics for a tracked on-chain wallet.

Stored in Redis at ``oracle:state:wallets``. Maintained by whale-detector (WADE).
Used for anomaly scoring and copy-trade eligibility decisions.
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import ClassVar, Optional
from enum import Enum
from datetime import datetime


class ReputationTier(str, Enum):
    SHARK    = "Shark"
    INFORMED = "Informed"
    UNKNOWN  = "Unknown"
    NOISE    = "Noise"


# Thresholds for algorithmic tier assignment (requires >= 10 trades)
SHARK_WIN_RATE    = 0.65
SHARK_POS_SIZE    = 5000.0
INFORMED_WIN_RATE = 0.55
INFORMED_POS_SIZE = 2000.0
NOISE_WIN_RATE    = 0.40
MIN_TRADES_TO_CLASSIFY = 10


class WalletProfile(BaseModel):
    wallet_address:             str
    reputation_tier:            ReputationTier = ReputationTier.UNKNOWN
    tier_assignment_method:     str = "algorithmic"  # "algorithmic" | "manual"
    historical_pnl_usd:         float = 0.0
    win_rate:                   float = 0.0
    typical_position_size_usd:  float = 0.0
    market_category_preference: list[str] = []
    first_seen_at:              datetime
    last_active_at:             datetime
    total_trades_tracked:       int = 0
    notes:                      Optional[str] = None

    # Redis state key (no channel — state only)
    STATE_KEY_PREFIX: ClassVar[str] = "oracle:state:wallets"  # HSET field = wallet_address
