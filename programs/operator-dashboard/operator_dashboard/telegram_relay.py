"""Task 9 — Telegram relay for operator alerts.

Subscribes to oracle:operator_alert. Forwards action_required and
warning alerts to Telegram with rate limiting (1 msg / 3s).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from oracle_shared.contracts.operator_alert import AlertSeverity, OperatorAlert

from operator_dashboard.config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_RATE_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)


class TelegramRelay:
    """Forward high-severity alerts to Telegram."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._running = False
        self._bot: Any = None

    async def start(self) -> None:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("TelegramRelay: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set -- skipping")
            return

        try:
            from telegram import Bot
            self._bot = Bot(token=TELEGRAM_BOT_TOKEN)
        except ImportError:
            logger.warning("TelegramRelay: python-telegram-bot not installed -- skipping")
            return

        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(OperatorAlert.CHANNEL)
        logger.info("TelegramRelay: subscribed to %s", OperatorAlert.CHANNEL)

        try:
            while self._running:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if msg is None:
                    await asyncio.sleep(0.05)
                    continue
                if msg["type"] != "message":
                    continue

                try:
                    alert = OperatorAlert.model_validate_json(msg["data"])
                    if alert.severity in (
                        AlertSeverity.ACTION_REQUIRED,
                        AlertSeverity.WARNING,
                    ):
                        await self._send(alert)
                        await asyncio.sleep(TELEGRAM_RATE_LIMIT_SECONDS)
                except Exception:
                    logger.warning("TelegramRelay: failed to process", exc_info=True)
        finally:
            await pubsub.unsubscribe(OperatorAlert.CHANNEL)
            await pubsub.aclose()

    async def _send(self, alert: OperatorAlert) -> None:
        """Send a Telegram message."""
        text = (
            f"{alert.title}\n\n"
            f"{alert.body}\n\n"
            f"Severity: {alert.severity.value}"
        )
        try:
            await self._bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text[:4096],  # Telegram message length limit
            )
            logger.info("TelegramRelay: sent alert %s", alert.alert_id[:8])
        except Exception:
            logger.warning("TelegramRelay: send failed", exc_info=True)

    async def stop(self) -> None:
        self._running = False
        logger.info("TelegramRelay: stopped")
