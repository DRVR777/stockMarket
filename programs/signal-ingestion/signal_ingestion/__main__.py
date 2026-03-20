"""signal-ingestion entry point.

Usage:  uv run python -m signal_ingestion
        (or via Makefile: make signal-ingestion)

Reads env from .env in the oracle root (loaded by python-dotenv).
Requires Redis to be running: make up

Starts all adapters concurrently. Each adapter runs its own async loop.
Graceful shutdown on SIGINT / SIGTERM stops all adapters then closes Redis.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

# Load .env from the oracle root (two levels up from programs/signal-ingestion/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

import redis.asyncio as aioredis

from signal_ingestion.config import (
    REDIS_URL,
    POLYMARKET_REST_POLL_INTERVAL,
    POLYMARKET_WS_MAX_MARKETS,
    POLYMARKET_WS_RECONNECT_DELAY,
    POLYGON_RECONNECT_DELAY,
    NEWSAPI_POLL_INTERVAL,
    WIKIPEDIA_POLL_INTERVAL,
    REDDIT_POLL_INTERVAL,
    BIRDEYE_RECONNECT_DELAY,
    BIRDEYE_REST_FALLBACK_INTERVAL,
    LOG_LEVEL,
    LOG_FORMAT,
)
from signal_ingestion.adapters.polymarket_rest import PolymarketRESTAdapter
from signal_ingestion.adapters.polymarket_ws import PolymarketWSAdapter
from signal_ingestion.adapters.polygon_onchain import PolygonOnchainAdapter
from signal_ingestion.adapters.newsapi_adapter import NewsAPIAdapter
from signal_ingestion.adapters.wikipedia_adapter import WikipediaAdapter
from signal_ingestion.adapters.reddit_adapter import RedditAdapter
from signal_ingestion.adapters.birdeye_ws import BirdeyeWSAdapter
from signal_ingestion.adapters.ai_opinion import AIOpinionAdapter

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("signal_ingestion")


async def main() -> None:
    """Start all signal-ingestion adapters and run until interrupted."""
    logger.info("signal-ingestion starting  redis=%s", REDIS_URL)

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

    # ── Instantiate all adapters ──────────────────────────────────────────────

    adapters = [
        PolymarketRESTAdapter(
            redis_client=redis_client,
            poll_interval=POLYMARKET_REST_POLL_INTERVAL,
        ),
        PolymarketWSAdapter(
            redis_client=redis_client,
            max_markets=POLYMARKET_WS_MAX_MARKETS,
            reconnect_delay=POLYMARKET_WS_RECONNECT_DELAY,
        ),
        PolygonOnchainAdapter(
            redis_client=redis_client,
            reconnect_delay=POLYGON_RECONNECT_DELAY,
        ),
        NewsAPIAdapter(
            redis_client=redis_client,
            poll_interval=NEWSAPI_POLL_INTERVAL,
        ),
        WikipediaAdapter(
            redis_client=redis_client,
            poll_interval=WIKIPEDIA_POLL_INTERVAL,
        ),
        RedditAdapter(
            redis_client=redis_client,
            poll_interval=REDDIT_POLL_INTERVAL,
        ),
        BirdeyeWSAdapter(
            redis_client=redis_client,
            reconnect_delay=BIRDEYE_RECONNECT_DELAY,
            rest_fallback_interval=BIRDEYE_REST_FALLBACK_INTERVAL,
        ),
        AIOpinionAdapter(
            redis_client=redis_client,
        ),
    ]

    # ── Graceful shutdown ─────────────────────────────────────────────────────

    loop = asyncio.get_running_loop()

    def _shutdown(sig_name: str) -> None:
        logger.info("signal-ingestion received %s — shutting down", sig_name)
        for adapter in adapters:
            asyncio.ensure_future(adapter.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals
            pass

    # ── Run all adapters concurrently ─────────────────────────────────────────

    try:
        await asyncio.gather(*(adapter.start() for adapter in adapters))
    finally:
        await redis_client.aclose()
        logger.info("signal-ingestion stopped")


if __name__ == "__main__":
    asyncio.run(main())
