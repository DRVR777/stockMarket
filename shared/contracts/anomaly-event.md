# Contract: AnomalyEvent

## Status
defined — build against this shape

## Produced By
whale-detector

## Consumed By
reasoning-engine, operator-dashboard

## Shape

```python
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from datetime import datetime

class AnomalyEvent(BaseModel):
    event_id:             str      = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:            datetime
    wallet_address:       str
    market_id:            str
    outcome:              str                      # "YES" | "NO" | specific option label
    notional_usd:         float                    # USD value of the flagged order fill
    anomaly_score:        float                    # 0.0–1.0
    wallet_profile:       Optional[dict] = None    # WalletProfile.model_dump() if wallet is in registry
    trigger_reasons:      list[str]                # e.g. ["large_order", "tier_1_wallet", "cascade_buy"]
    copy_trade_eligible:  bool                     # anomaly_score >= operator threshold
    cascade_wallets:      Optional[list[str]] = None  # other wallet addresses in same cascade window
    source_signal_id:     str                      # the polygon_clob Signal that triggered this
```

## trigger_reasons values
- `large_order` — notional_usd >= configured threshold
- `tier_1_wallet` — wallet_profile.reputation_tier == "Shark"
- `tier_2_wallet` — wallet_profile.reputation_tier == "Informed"
- `cascade_buy` — N+ wallets bought same outcome within T seconds
- `size_deviation` — order is >2x wallet's typical_position_size_usd
- `pre_resolution` — market resolves within configured window (e.g. 7 days)

## Redis channel
`oracle:anomaly_event`

## Serialization
`event.model_dump_json()` → publish. `AnomalyEvent.model_validate_json(msg)` → consume.
