"""osint-fusion entry point.

Usage:  uv run python -m osint_fusion
        (or via Makefile: make osint-fusion)

Reads env from .env in the oracle root (loaded by python-dotenv).
Requires Redis to be running: make up

Pipeline:
  1. Initialize ChromaDB, populate from Redis market state
  2. Subscribe to oracle:signal — extract text, embed, search, score, emit Insight
  3. Update MarketState in Redis with new Insights + Haiku summary
  4. Subscribe to oracle:post_mortem — update credibility weights

Graceful shutdown on SIGINT / SIGTERM.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

# Load .env from the oracle root (two levels up from programs/osint-fusion/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

import redis.asyncio as aioredis

from oracle_shared.contracts.signal import Signal

from osint_fusion.config import LOG_FORMAT, LOG_LEVEL, REDIS_URL
from osint_fusion.chroma_store import ChromaStore
from osint_fusion.embedder import Embedder
from osint_fusion.credibility import CredibilityWeighter
from osint_fusion.insight_emitter import InsightEmitter
from osint_fusion.market_state_updater import MarketStateUpdater
from osint_fusion.signal_subscriber import SignalSubscriber
from osint_fusion.postmortem_subscriber import PostMortemSubscriber

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("osint_fusion")


class OSFEPipeline:
    """Orchestrates the full OSFE processing pipeline for a single signal."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        chroma: ChromaStore,
        embedder: Embedder,
        credibility: CredibilityWeighter,
        insight_emitter: InsightEmitter,
        market_updater: MarketStateUpdater,
    ) -> None:
        self.chroma = chroma
        self.embedder = embedder
        self.credibility = credibility
        self.insight_emitter = insight_emitter
        self.market_updater = market_updater

    async def process(self, sig: Signal, raw_text: str) -> None:
        """Run a signal through the full OSFE pipeline (Steps 3-7)."""
        # Step 3: Embed
        embedding = await self.embedder.embed_single(raw_text)

        # Step 4: Similarity search
        similarity_scores = self.chroma.query(embedding)
        if not similarity_scores:
            logger.debug(
                "OSFE: no market matches for signal %s", sig.signal_id
            )
            return

        # Step 5: Credibility weight
        weight = await self.credibility.get_weight(sig.category.value)

        # Step 6: Emit Insight
        insight = await self.insight_emitter.emit(
            source_signal_id=sig.signal_id,
            source_category=sig.category.value,
            raw_text=raw_text,
            similarity_scores=similarity_scores,
            credibility_weight=weight,
        )

        # Step 7: Update MarketState for each matched market
        await self.market_updater.update(insight)


async def main() -> None:
    """Start osint-fusion and run until interrupted."""
    logger.info("osint-fusion (OSFE) starting  redis=%s", REDIS_URL)

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Step 1: ChromaDB setup
    embedder = Embedder()
    chroma = ChromaStore()
    market_count = await chroma.populate_from_redis(redis_client, embedder.embed)
    logger.info("OSFE: ChromaDB populated with %d markets", market_count)

    # Pipeline components
    credibility = CredibilityWeighter(redis_client)
    insight_emitter = InsightEmitter(redis_client)
    market_updater = MarketStateUpdater(redis_client)

    pipeline = OSFEPipeline(
        redis_client=redis_client,
        chroma=chroma,
        embedder=embedder,
        credibility=credibility,
        insight_emitter=insight_emitter,
        market_updater=market_updater,
    )

    # Step 2: Signal subscriber
    signal_sub = SignalSubscriber(redis_client, on_signal=pipeline.process)

    # Step 8: PostMortem subscriber
    pm_sub = PostMortemSubscriber(redis_client, credibility)

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    components = [signal_sub, pm_sub]

    def _shutdown(sig_name: str) -> None:
        logger.info("osint-fusion received %s — shutting down", sig_name)
        for comp in components:
            asyncio.ensure_future(comp.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
        except NotImplementedError:
            pass

    # Run signal subscriber + postmortem subscriber concurrently
    try:
        await asyncio.gather(
            signal_sub.start(),
            pm_sub.start(),
        )
    finally:
        await redis_client.aclose()
        logger.info("osint-fusion (OSFE) stopped")


if __name__ == "__main__":
    asyncio.run(main())
