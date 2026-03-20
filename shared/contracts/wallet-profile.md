# Contract: WalletProfile

## Status
defined — build against this shape

## Produced By
whale-detector (maintains the live registry in Redis)

## Consumed By
knowledge-base (persists to /wallets/ vault), embedded in AnomalyEvent

## Shape

```python
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from datetime import datetime

class ReputationTier(str, Enum):
    SHARK    = "Shark"      # consistently profitable, large positions, early movers
    INFORMED = "Informed"   # profitable but smaller or less consistent
    UNKNOWN  = "Unknown"    # insufficient history to classify
    NOISE    = "Noise"      # historically unprofitable or random

class WalletProfile(BaseModel):
    wallet_address:               str
    reputation_tier:              ReputationTier = ReputationTier.UNKNOWN
    tier_assignment_method:       str = "algorithmic"  # "algorithmic" | "manual"
    historical_pnl_usd:           float = 0.0
    win_rate:                     float = 0.0     # 0.0–1.0; requires >= 10 trades to be meaningful
    typical_position_size_usd:    float = 0.0     # rolling median of last 20 fills
    market_category_preference:   list[str] = []  # e.g. ["politics", "crypto", "sports"]
    first_seen_at:                datetime
    last_active_at:               datetime
    total_trades_tracked:         int = 0
    notes:                        Optional[str] = None
```

## Reputation tier thresholds (algorithmic assignment)
Applied after >= 10 trades tracked:
| Tier | Criteria |
|------|---------|
| Shark | win_rate >= 0.65 AND typical_position_size_usd >= 5000 |
| Informed | win_rate >= 0.55 OR typical_position_size_usd >= 2000 |
| Noise | win_rate < 0.40 |
| Unknown | < 10 trades OR criteria not met |

## Storage
Redis hash `oracle:state:wallets`: field = wallet_address, value = `profile.model_dump_json()`.
KBPM writes the full profile to `/wallets/{wallet_address}.md` periodically (on each update to the wallet's document).

## Serialization
`profile.model_dump_json()` → Redis hash value. `WalletProfile.model_validate_json(val)` → consume.
No Redis pub/sub channel — WalletProfile is state (read from Redis), not an event.
It is embedded inside AnomalyEvent when a flagged wallet is found in the registry.
