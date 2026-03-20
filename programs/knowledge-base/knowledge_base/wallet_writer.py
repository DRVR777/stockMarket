"""Task 5 — WalletWriter.

Subscribes to oracle:anomaly_event. Writes/overwrites vault/wallets/{address}.md
with the full WalletProfile + event history.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from oracle_shared.contracts.anomaly_event import AnomalyEvent

from knowledge_base.vault import vault_path, write_md, read_md, append_section

logger = logging.getLogger(__name__)


class WalletWriter:
    """Subscribe to AnomalyEvent and maintain wallet vault files."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._running = False

    async def start(self) -> None:
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(AnomalyEvent.CHANNEL)
        logger.info("WalletWriter: subscribed to %s", AnomalyEvent.CHANNEL)

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
                    event = AnomalyEvent.model_validate_json(msg["data"])
                    self._process(event)
                except Exception:
                    logger.warning("WalletWriter: failed to process", exc_info=True)
        finally:
            await pubsub.unsubscribe(AnomalyEvent.CHANNEL)
            await pubsub.aclose()

    def _process(self, event: AnomalyEvent) -> None:
        """Write or update wallet vault file."""
        if not event.wallet_profile:
            return

        addr = event.wallet_address
        path = vault_path("wallets", f"{addr}.md")

        profile_json = json.dumps(event.wallet_profile, indent=2, default=str)

        front_matter = {
            "wallet_address": addr,
            "reputation_tier": event.wallet_profile.get("reputation_tier", "Unknown"),
            "total_trades": event.wallet_profile.get("total_trades_tracked", 0),
        }

        # Build body with profile + append event
        event_entry = (
            f"- [{event.timestamp.isoformat()[:19]}] "
            f"${event.notional_usd:,.0f} {event.outcome} "
            f"on {event.market_id[:16]}… "
            f"score={event.anomaly_score:.2f}\n"
        )

        if path.exists():
            # Append event to existing file
            append_section(path, "Events", event_entry)
            logger.debug("WalletWriter: appended event to %s", path.name)
        else:
            body = (
                f"# Wallet {addr[:10]}...\n\n"
                f"## Profile\n```json\n{profile_json}\n```\n\n"
                f"## Events\n{event_entry}"
            )
            write_md(path, front_matter, body)
            logger.info("WalletWriter: created %s", path.name)

    async def stop(self) -> None:
        self._running = False
        logger.info("WalletWriter: stopped")
