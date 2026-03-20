# Contract: Signal

## Status
defined — build against this shape

## Produced By
signal-ingestion

## Consumed By
whale-detector, osint-fusion

## Shape

```python
from pydantic import BaseModel, Field
from typing import Any, Optional
from enum import Enum
import uuid
from datetime import datetime

class SignalCategory(str, Enum):
    ON_CHAIN  = "on_chain"
    NEWS      = "news"
    SOCIAL    = "social"
    PRICE     = "price"
    AI        = "ai_generated"

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
    raw_payload: dict[str, Any]          # raw data from source — structure varies by source_id
    confidence:  Optional[float] = None  # 0.0–1.0 where applicable; null otherwise
    market_ids:  Optional[list[str]] = None  # pre-associated Polymarket market IDs if SIL can determine them
```

## Payload shape by source_id

| source_id | raw_payload keys |
|-----------|-----------------|
| `polymarket_rest` | `market_id`, `question`, `outcome_prices: {YES: float, NO: float}`, `volume_usd`, `liquidity_usd`, `end_date` |
| `polymarket_ws` | `market_id`, `outcome`, `price`, `side` (buy/sell), `size` |
| `polygon_clob` | `tx_hash`, `wallet`, `market_id`, `outcome`, `side`, `price`, `size_usd`, `block_number`, `block_timestamp` |
| `newsapi` | `title`, `description`, `url`, `source_name`, `published_at`, `query_used` |
| `wikipedia` | `page_title`, `summary`, `edit_timestamp`, `edit_comment`, `diff_url` |
| `reddit` | `subreddit`, `post_id`, `title`, `score`, `num_comments`, `created_utc`, `url` |
| `birdeye` | `token_address`, `symbol`, `price_usd`, `price_change_24h_pct`, `volume_24h_usd` |
| `ai_opinion` | `model`, `prompt_used`, `response_text`, `market_ids_queried: list[str]` |

## Redis channel
`oracle:signal`

## Serialization
`signal.model_dump_json()` → publish. `Signal.model_validate_json(msg)` → consume.
