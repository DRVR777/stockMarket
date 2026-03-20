"""OperatorAlert contract — notifications pushed to the operator via Telegram and the dashboard.

Published to ``oracle:operator_alert``. Produced by whale-detector, reasoning-engine,
and system-level circuit breakers. Consumed by operator-dashboard.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import ClassVar, Optional
from enum import Enum
from datetime import datetime
import uuid

# Maximum character length for alert titles (used as Telegram message header)
ALERT_TITLE_MAX_LENGTH = 80


class AlertType(str, Enum):
    ANOMALY         = "anomaly"
    THESIS          = "thesis"
    CIRCUIT_BREAKER = "circuit_breaker"
    SYSTEM          = "system"


class AlertSeverity(str, Enum):
    INFO            = "info"
    WARNING         = "warning"
    ACTION_REQUIRED = "action_required"


class OperatorAlert(BaseModel):
    alert_id:        str           = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at:      datetime
    alert_type:      AlertType
    severity:        AlertSeverity
    title:           str           # max ALERT_TITLE_MAX_LENGTH chars — Telegram header
    body:            str
    action_required: bool
    action_options:  Optional[list[str]] = None  # e.g. ["approve_copy_trade", "dismiss"]
    linked_event_id: Optional[str]       = None
    acknowledged:    bool = False
    acknowledged_at: Optional[datetime]  = None
    action_taken:    Optional[str]       = None

    # Redis channel
    CHANNEL: ClassVar[str] = "oracle:operator_alert"
