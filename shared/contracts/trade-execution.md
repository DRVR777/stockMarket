# Contract: TradeExecution

## Status
defined — build against this shape

## Produced By
signal-ingestion (Polymarket execution path), solana-executor

## Consumed By
knowledge-base

## Shape

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import uuid
from datetime import datetime

class MarketType(str, Enum):
    POLYMARKET = "polymarket"
    SOLANA     = "solana"

class ExecutionSource(str, Enum):
    RE_THESIS         = "re_thesis"
    COPY_TRADE        = "copy_trade"
    SOE_MEAN_REVERSION = "soe_mean_reversion"

class ExecutionStatus(str, Enum):
    OPEN   = "open"
    CLOSED = "closed"
    FAILED = "failed"

class ExitReason(str, Enum):
    TAKE_PROFIT      = "take_profit"
    STOP_LOSS        = "stop_loss"
    MARKET_RESOLVED  = "market_resolved"
    MANUAL           = "manual"

class TradeExecution(BaseModel):
    execution_id:          str            = Field(default_factory=lambda: str(uuid.uuid4()))
    thesis_id:             Optional[str]  = None   # linked TradeThesis; null for copy_trade with no RE thesis
    market_id:             str
    market_type:           MarketType
    direction:             str            # "buy" | "sell"
    outcome:               Optional[str]  = None   # Polymarket: "YES"/"NO"; Solana: null
    entry_price:           float
    size_usd:              float
    executed_at:           datetime
    execution_source:      ExecutionSource
    status:                ExecutionStatus = ExecutionStatus.OPEN
    exit_price:            Optional[float]    = None
    exit_at:               Optional[datetime] = None
    exit_reason:           Optional[ExitReason] = None
    realized_pnl_usd:      Optional[float]    = None
    circuit_breaker_checked: bool = True
    tx_hash:               Optional[str]      = None   # on-chain tx hash (Polygon or Solana)
    copy_trade_source_wallet: Optional[str]   = None   # set if execution_source == copy_trade
```

## Storage
Redis hash `oracle:state:positions`: field = execution_id, value = `execution.model_dump_json()`.
Updated in-place on close (status, exit_price, exit_at, exit_reason, realized_pnl_usd populated).

## Redis channel
`oracle:trade_execution` — published on open AND on close (status update).

## Serialization
`execution.model_dump_json()` → publish + hash value. `TradeExecution.model_validate_json(msg)` → consume.
