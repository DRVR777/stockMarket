"""Steps 6 & 7 — AnomalyEvent and OperatorAlert emission.

Step 6: Assemble an ``AnomalyEvent`` from all prior pipeline outputs. Set
``copy_trade_eligible`` based on the operator-configurable threshold.
Publish to ``AnomalyEvent.CHANNEL``.

Step 7: If ``copy_trade_eligible``, create an ``OperatorAlert`` with
``alert_type=anomaly, severity=action_required`` and action options for
the operator. Publish to ``OperatorAlert.CHANNEL``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from oracle_shared.contracts.anomaly_event import AnomalyEvent
from oracle_shared.contracts.operator_alert import (
    ALERT_TITLE_MAX_LENGTH,
    AlertSeverity,
    AlertType,
    OperatorAlert,
)
from oracle_shared.contracts.wallet_profile import WalletProfile

from whale_detector.config import (
    COPY_TRADE_THRESHOLD_DEFAULT,
    COPY_TRADE_THRESHOLD_PARAM_KEY,
    PARAMS_STATE_KEY,
)

logger = logging.getLogger(__name__)


class EventEmitter:
    """Assemble and publish AnomalyEvent and OperatorAlert to Redis."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def emit(
        self,
        wallet_address: str,
        market_id: str,
        outcome: str,
        notional_usd: float,
        anomaly_score: float,
        wallet_profile: WalletProfile,
        trigger_reasons: list[str],
        source_signal_id: str,
        cascade_wallets: Optional[list[str]] = None,
    ) -> AnomalyEvent:
        """Build and publish an AnomalyEvent; optionally emit an OperatorAlert.

        Returns the published AnomalyEvent.
        """
        # Step 6: determine copy-trade eligibility
        copy_threshold = await self._get_copy_trade_threshold()
        copy_trade_eligible = anomaly_score >= copy_threshold

        event = AnomalyEvent(
            timestamp=datetime.now(timezone.utc),
            wallet_address=wallet_address,
            market_id=market_id,
            outcome=outcome,
            notional_usd=notional_usd,
            anomaly_score=anomaly_score,
            wallet_profile=wallet_profile.model_dump(),
            trigger_reasons=trigger_reasons,
            copy_trade_eligible=copy_trade_eligible,
            cascade_wallets=cascade_wallets,
            source_signal_id=source_signal_id,
        )

        try:
            await self._redis.publish(
                AnomalyEvent.CHANNEL, event.model_dump_json(),
            )
        except (ConnectionError, OSError) as exc:
            logger.error("EventEmitter: Redis publish failed for AnomalyEvent %s: %s", event.event_id, exc)
            return event

        logger.info(
            "EventEmitter: published AnomalyEvent %s  score=%.3f  copy_eligible=%s",
            event.event_id,
            anomaly_score,
            copy_trade_eligible,
        )

        # Step 7: emit OperatorAlert if copy-trade eligible
        if copy_trade_eligible:
            await self._emit_operator_alert(event)

        return event

    async def _emit_operator_alert(self, event: AnomalyEvent) -> None:
        """Create and publish an OperatorAlert for a copy-trade-eligible event."""
        title = (
            f"Whale alert: ${event.notional_usd:,.0f} on {event.market_id}"
        )
        # Truncate title to contract maximum
        if len(title) > ALERT_TITLE_MAX_LENGTH:
            title = title[: ALERT_TITLE_MAX_LENGTH - 3] + "..."

        body_lines = [
            f"Wallet: {event.wallet_address}",
            f"Outcome: {event.outcome}",
            f"Anomaly score: {event.anomaly_score:.3f}",
            f"Trigger reasons: {', '.join(event.trigger_reasons)}",
        ]
        if event.cascade_wallets:
            body_lines.append(
                f"Cascade wallets ({len(event.cascade_wallets)}): "
                + ", ".join(event.cascade_wallets[:5]),
            )

        alert = OperatorAlert(
            created_at=datetime.now(timezone.utc),
            alert_type=AlertType.ANOMALY,
            severity=AlertSeverity.ACTION_REQUIRED,
            title=title,
            body="\n".join(body_lines),
            action_required=True,
            action_options=["approve_copy_trade", "dismiss"],
            linked_event_id=event.event_id,
        )

        await self._redis.publish(
            OperatorAlert.CHANNEL, alert.model_dump_json(),
        )
        logger.info(
            "EventEmitter: published OperatorAlert %s for event %s",
            alert.alert_id,
            event.event_id,
        )

    async def _get_copy_trade_threshold(self) -> float:
        """Read copy-trade threshold from Redis params, with default fallback."""
        raw: Optional[str] = await self._redis.hget(
            PARAMS_STATE_KEY, COPY_TRADE_THRESHOLD_PARAM_KEY,
        )
        if raw is not None:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
        return COPY_TRADE_THRESHOLD_DEFAULT
