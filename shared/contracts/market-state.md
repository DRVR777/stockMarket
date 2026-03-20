# Contract: MarketState

## Status
defined — build against this shape

## Produced By
osint-fusion (rolling semantic state); signal-ingestion updates price/liquidity fields

## Consumed By
reasoning-engine

## Shape

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class MarketState(BaseModel):
    market_id:               str
    market_question:         str
    current_price_yes:       float          # 0.0–1.0 (implied probability of YES)
    resolution_deadline:     datetime
    liquidity_usd:           float
    last_price_updated:      datetime       # when SIL last updated price fields
    last_signal_at:          Optional[datetime] = None  # when OSFE last associated a signal
    recent_insights:         list[dict]     # last 20 Insight.model_dump() objects, newest first
    semantic_state_summary:  str            # OSFE's rolling synthesis of all recent signals
    signal_count_24h:        int = 0        # number of signals associated in last 24h
    whale_event_count_24h:   int = 0        # number of AnomalyEvents for this market in last 24h
```

## Storage
MarketState is stored in Redis at `oracle:state:markets` hash: field = market_id, value = `state.model_dump_json()`.
Published to `oracle:market_state` channel on every update.

## recent_insights window
Maximum 20 Insight objects. When a 21st arrives, drop the oldest. RE loads this from Redis directly — it is not replayed from the event channel.

## Redis channel
`oracle:market_state`

## Serialization
`state.model_dump_json()` → hash value + publish. `MarketState.model_validate_json(msg)` → consume.
