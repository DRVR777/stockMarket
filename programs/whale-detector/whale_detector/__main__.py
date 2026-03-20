"""whale-detector entry point.

Usage:  uv run python -m whale_detector
        (or via Makefile: make whale-detector)

Reads env from .env in the oracle root (loaded by python-dotenv).
Requires Redis to be running: make up

Subscribes to ``oracle:signal`` and runs the full WADE pipeline:
  1. Filter on-chain polygon_clob signals
  2. Threshold flag large orders
  3. Wallet registry lookup / create
  4. Anomaly score (size, wallet, time)
  5. Cascade detection
  6. Publish AnomalyEvent
  7. Publish OperatorAlert (if copy-trade eligible)
  8. Update WalletProfile

Graceful shutdown on SIGINT / SIGTERM.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

# Load .env from the oracle root (two levels up from programs/whale-detector/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

import redis.asyncio as aioredis

from oracle_shared.contracts.signal import Signal

from whale_detector.config import LOG_FORMAT, LOG_LEVEL, REDIS_URL
from whale_detector.signal_subscriber import SignalSubscriber
from whale_detector.threshold_flagger import ThresholdFlagger
from whale_detector.wallet_registry import WalletRegistry
from whale_detector.anomaly_scorer import AnomalyScorer
from whale_detector.cascade_detector import CascadeDetector
from whale_detector.event_emitter import EventEmitter

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("whale_detector")


class WhalePipeline:
    """Orchestrates the full WADE detection pipeline for a single signal."""

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self.threshold_flagger = ThresholdFlagger(redis_client)
        self.wallet_registry = WalletRegistry(redis_client)
        self.anomaly_scorer = AnomalyScorer(redis_client)
        self.cascade_detector = CascadeDetector(redis_client)
        self.event_emitter = EventEmitter(redis_client)

    async def process(self, sig: Signal) -> None:
        """Run a signal through the full WADE pipeline (Steps 2-8)."""
        # Step 2: Threshold check
        flagged = await self.threshold_flagger.evaluate(sig)
        if not flagged:
            return

        payload = sig.raw_payload
        wallet_address: str = payload.get("wallet", "")
        market_id: str = payload.get("market_id", "")
        outcome: str = payload.get("outcome", "")
        size_usd: float = float(payload.get("size_usd", 0.0))

        # Step 3: Wallet registry lookup
        profile = await self.wallet_registry.get_or_create(wallet_address)

        # Step 4: Anomaly scoring (reads market state from Redis internally)
        anomaly_score, trigger_reasons = await self.anomaly_scorer.score(
            size_usd=size_usd,
            market_id=market_id,
            wallet_profile=profile,
        )

        # Step 5: Cascade detection
        is_cascade, cascade_wallets = await self.cascade_detector.check(
            market_id=market_id,
            outcome=outcome,
            wallet_address=wallet_address,
        )
        if is_cascade:
            trigger_reasons.append("cascade_buy")

        # Step 6 & 7: Emit AnomalyEvent (and OperatorAlert if eligible)
        await self.event_emitter.emit(
            wallet_address=wallet_address,
            market_id=market_id,
            outcome=outcome,
            notional_usd=size_usd,
            anomaly_score=anomaly_score,
            wallet_profile=profile,
            trigger_reasons=trigger_reasons,
            source_signal_id=sig.signal_id,
            cascade_wallets=cascade_wallets if is_cascade else None,
        )

        # Step 8: Update wallet profile
        await self.wallet_registry.update_after_event(profile, size_usd)


async def main() -> None:
    """Start the whale-detector and run until interrupted."""
    # Mask credentials in URL for logging
    _safe_url = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
    logger.info("whale-detector (WADE) starting  redis=%s", _safe_url)

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    pipeline = WhalePipeline(redis_client)

    subscriber = SignalSubscriber(
        redis_client=redis_client,
        on_signal=pipeline.process,
    )

    # Graceful shutdown
    loop = asyncio.get_running_loop()

    def _shutdown(sig_name: str) -> None:
        logger.info("whale-detector received %s — shutting down", sig_name)
        asyncio.ensure_future(subscriber.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals
            pass

    try:
        await subscriber.start()
    finally:
        await redis_client.aclose()
        logger.info("whale-detector (WADE) stopped")


if __name__ == "__main__":
    asyncio.run(main())
