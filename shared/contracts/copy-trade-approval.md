# Contract: CopyTradeApproval

## Status
defined — build against this shape

## Produced By
operator-dashboard (on operator approval action)

## Consumed By
signal-ingestion (execution path)

## Shape

```python
from pydantic import BaseModel
from datetime import datetime

class CopyTradeApproval(BaseModel):
    approval_id:       str       # matches alert_id from OperatorAlert
    anomaly_event_id:  str       # the AnomalyEvent being approved for copy
    approved_at:       datetime
    operator_note:     str = ""  # optional operator comment
```

## Redis channel
`oracle:copy_trade_approved` — signal-ingestion subscribes and executes copy trade on receipt.
Message payload: `approval.model_dump_json()`

## Note
This contract closes the inference logged in pending.txt 2026-03-14T01:00:00Z.
Add to shared/MANIFEST.md contracts table.
