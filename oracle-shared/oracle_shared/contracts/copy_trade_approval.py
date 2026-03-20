"""CopyTradeApproval contract — operator approval for executing a copy-trade based on an anomaly event.

Published to ``oracle:copy_trade_approved``. Produced by operator-dashboard,
consumed by signal-ingestion to trigger the copy-trade execution flow.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import ClassVar
from datetime import datetime
import uuid


class CopyTradeApproval(BaseModel):
    approval_id:      str = Field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_event_id: str
    approved_at:      datetime
    operator_note:    str = ""

    # Redis channel — signal-ingestion subscribes to execute the copy trade
    CHANNEL: ClassVar[str] = "oracle:copy_trade_approved"
