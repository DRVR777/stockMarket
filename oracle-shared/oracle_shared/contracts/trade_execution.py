"""TradeExecution contract — represents a single position entry or exit across Polymarket or Solana.

Published to ``oracle:trade_execution``. Produced by solana-executor (SOE) and
the Polymarket execution path. Consumed by knowledge-base for post-mortem generation.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import ClassVar, Optional
from enum import Enum
from datetime import datetime
import uuid


class MarketType(str, Enum):
    POLYMARKET = "polymarket"
    SOLANA     = "solana"


class ExecutionSource(str, Enum):
    RE_THESIS          = "re_thesis"
    COPY_TRADE         = "copy_trade"
    SOE_MEAN_REVERSION = "soe_mean_reversion"


class ExecutionStatus(str, Enum):
    OPEN   = "open"
    CLOSED = "closed"
    FAILED = "failed"


class ExitReason(str, Enum):
    TAKE_PROFIT     = "take_profit"
    STOP_LOSS       = "stop_loss"
    MARKET_RESOLVED = "market_resolved"
    MANUAL          = "manual"


class TradeExecution(BaseModel):
    execution_id:             str            = Field(default_factory=lambda: str(uuid.uuid4()))
    thesis_id:                Optional[str]  = None
    market_id:                str
    market_type:              MarketType
    direction:                str            # "buy" | "sell"
    outcome:                  Optional[str]  = None  # Polymarket: "YES"/"NO"; Solana: null
    entry_price:              float
    size_usd:                 float
    executed_at:              datetime
    execution_source:         ExecutionSource
    status:                   ExecutionStatus = ExecutionStatus.OPEN
    exit_price:               Optional[float]    = None
    exit_at:                  Optional[datetime] = None
    exit_reason:              Optional[ExitReason] = None
    realized_pnl_usd:         Optional[float]    = None
    circuit_breaker_checked:  bool = True
    tx_hash:                  Optional[str] = None
    copy_trade_source_wallet: Optional[str] = None

    # Redis
    CHANNEL: ClassVar[str] = "oracle:trade_execution"
    STATE_KEY_PREFIX: ClassVar[str] = "oracle:state:positions"  # HSET field = execution_id
