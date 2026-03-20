# Contract: Insight

## Status
defined — build against this shape

## Produced By
osint-fusion

## Consumed By
reasoning-engine

## Shape

```python
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from datetime import datetime

class Insight(BaseModel):
    insight_id:                str            = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:                 datetime
    source_signal_id:          str            # Signal.signal_id that produced this
    source_category:           str            # mirrors Signal.category
    associated_market_ids:     list[str]      # markets matched via semantic similarity
    similarity_scores:         dict[str, float]  # {market_id: score 0.0–1.0}
    semantic_summary:          str            # OSFE's one-paragraph interpretation of the signal
    source_credibility_weight: float          # applied trust weight (0.0–2.0); default 1.0; >1.0 amplifies
    raw_text:                  str            # the text that was embedded (title + description, or post title, etc.)
```

## similarity_scores threshold
Only markets with similarity_score >= 0.65 are included in associated_market_ids.
Markets with score < 0.65 are dropped silently.

## source_credibility_weight defaults by category
| category | default weight |
|----------|---------------|
| on_chain | 1.5 |
| news | 1.0 |
| social | 0.6 |
| price | 1.2 |
| ai_generated | 0.8 |

Weights are updated by OSFE when PostMortem.source_weight_updates are received.

## Redis channel
`oracle:insight`

## Serialization
`insight.model_dump_json()` → publish. `Insight.model_validate_json(msg)` → consume.
