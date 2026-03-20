# Contract: PostMortem

## Status
defined — build against this shape

## Produced By
knowledge-base

## Consumed By
osint-fusion (source weight updates), operator-dashboard (post-mortem feed)

## Shape

```python
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from datetime import datetime

class SignalSummary(BaseModel):
    signal_id:  str
    category:   str
    summary:    str   # one sentence from raw_text or semantic_summary
    was_useful: Optional[bool] = None  # RE's retrospective judgement

class PostMortem(BaseModel):
    postmortem_id:                 str           = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at:                  datetime
    market_id:                     str
    market_question:               str
    thesis_id:                     Optional[str] = None
    execution_id:                  Optional[str] = None
    market_resolved_as:            str           # e.g. "YES", "NO", "VOID", or Solana asset symbol
    thesis_was_correct:            Optional[bool] = None  # null if no thesis was generated
    realized_pnl_usd:             Optional[float] = None
    signals_present:               list[SignalSummary]
    what_the_thesis_said:          str           # prose summary of RE's prediction
    what_happened:                 str           # prose summary of actual resolution
    what_would_have_changed_outcome: str         # RE's counterfactual analysis
    source_weight_updates:         dict[str, float]  # {source_id: delta} e.g. {"newsapi": +0.05, "reddit": -0.02}
    vault_path:                    str           # full path to markdown doc in KBPM vault
```

## source_weight_updates semantics
Deltas are applied to OSFE's credibility weights per source_id.
Positive delta: this source was more useful than average → amplify its weight.
Negative delta: this source was misleading or irrelevant → reduce its weight.
OSFE applies updates atomically when PostMortem is received.

## Redis channel
`oracle:post_mortem`

## Serialization
`pm.model_dump_json()` → publish. `PostMortem.model_validate_json(msg)` → consume.
