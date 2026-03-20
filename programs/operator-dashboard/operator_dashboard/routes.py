"""Tasks 7 & 8 — HTTP endpoints for actions and parameter control.

POST /action — handle copy-trade approval and dismiss actions
GET  /params — read all operator-configurable params
POST /params — update params in Redis
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from oracle_shared.contracts.copy_trade_approval import CopyTradeApproval

from operator_dashboard.config import PARAMS_KEY

logger = logging.getLogger(__name__)

router = APIRouter()


class ActionRequest(BaseModel):
    action: str  # "approve_copy_trade" | "dismiss"
    alert_id: str
    anomaly_event_id: str = ""


class ParamsUpdate(BaseModel):
    params: dict[str, str]


@router.post("/action")
async def handle_action(body: ActionRequest, request: Request) -> dict:
    """Handle operator actions (copy-trade approval, dismiss)."""
    redis = request.app.state.redis

    if body.action == "approve_copy_trade":
        # Publish copy-trade approval
        approval = CopyTradeApproval(
            anomaly_event_id=body.anomaly_event_id,
            approved_at=datetime.now(timezone.utc),
        )
        await redis.publish(
            CopyTradeApproval.CHANNEL,
            approval.model_dump_json(),
        )
        # Mark alert acknowledged
        await redis.hset(
            f"oracle:state:alerts:{body.alert_id}",
            "acknowledged", "true",
        )
        logger.info("Action: approved copy trade for event %s", body.anomaly_event_id)
        return {"status": "approved", "approval_id": approval.approval_id}

    elif body.action == "dismiss":
        await redis.hset(
            f"oracle:state:alerts:{body.alert_id}",
            "acknowledged", "true",
        )
        logger.info("Action: dismissed alert %s", body.alert_id)
        return {"status": "dismissed"}

    return {"status": "unknown_action"}


@router.get("/params")
async def get_params(request: Request) -> dict:
    """Read all operator-configurable params from Redis."""
    redis = request.app.state.redis
    raw = await redis.hgetall(PARAMS_KEY)
    return {"params": raw or {}}


@router.post("/params")
async def update_params(body: ParamsUpdate, request: Request) -> dict:
    """Update operator params in Redis."""
    redis = request.app.state.redis
    for key, value in body.params.items():
        await redis.hset(PARAMS_KEY, key, value)
    logger.info("Params updated: %s", list(body.params.keys()))
    return {"status": "updated", "count": len(body.params)}
