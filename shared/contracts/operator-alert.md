# Contract: OperatorAlert

## Status
defined — build against this shape

## Produced By
whale-detector (anomaly alerts, copy-trade prompts), reasoning-engine (thesis alerts, circuit breaker events)

## Consumed By
operator-dashboard (display + Telegram relay)

## Shape

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import uuid
from datetime import datetime

class AlertType(str, Enum):
    ANOMALY         = "anomaly"
    THESIS          = "thesis"
    CIRCUIT_BREAKER = "circuit_breaker"
    SYSTEM          = "system"

class AlertSeverity(str, Enum):
    INFO           = "info"
    WARNING        = "warning"
    ACTION_REQUIRED = "action_required"

class OperatorAlert(BaseModel):
    alert_id:         str            = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at:       datetime
    alert_type:       AlertType
    severity:         AlertSeverity
    title:            str            # short (< 80 chars) — used as Telegram message header
    body:             str            # full alert text with context
    action_required:  bool
    action_options:   Optional[list[str]] = None  # e.g. ["copy_trade", "dismiss"] | ["approve", "reject", "dismiss"]
    linked_event_id:  Optional[str]       = None  # thesis_id | execution_id | event_id
    acknowledged:     bool = False
    acknowledged_at:  Optional[datetime]  = None
    action_taken:     Optional[str]       = None  # which action_option was chosen
```

## alert_type → severity mapping (defaults)
| alert_type | severity |
|-----------|---------|
| anomaly (copy_trade_eligible=true) | action_required |
| anomaly (copy_trade_eligible=false) | info |
| thesis (decision=execute) | action_required |
| thesis (decision=flag_for_review) | warning |
| circuit_breaker | warning |
| system (error) | warning |
| system (startup/info) | info |

## action_options by context
- Copy-trade alert: `["approve_copy_trade", "dismiss"]`
- Thesis review: `["approve_execute", "reject", "dismiss"]`
- Circuit breaker: `["acknowledge", "reset_circuit_breaker"]`

## Redis channel
`oracle:operator_alert`

## Serialization
`alert.model_dump_json()` → publish. `OperatorAlert.model_validate_json(msg)` → consume.
Acknowledgement written back to `oracle:state:alerts:{alert_id}` in Redis.
