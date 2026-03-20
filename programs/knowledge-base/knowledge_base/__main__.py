"""knowledge-base entry point.

Usage:  uv run python -m knowledge_base
        (or via Makefile: make knowledge-base)

Runs:
  - Vault initializer (creates directory tree)
  - ThesisWriter (subscribe trade_thesis -> vault + ChromaDB)
  - WalletWriter (subscribe anomaly_event -> vault)
  - ExecutionSubscriber (subscribe trade_execution -> vault + post-mortem trigger)
  - MarketWriter (called by thesis/execution writers)
  - PostMortemGenerator (triggered on execution close -> Claude -> vault + publish)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

import redis.asyncio as aioredis

from oracle_shared.db import init_db

from knowledge_base.config import LOG_FORMAT, LOG_LEVEL, REDIS_URL
from knowledge_base.vault import init_vault
from knowledge_base.thesis_writer import ThesisWriter
from knowledge_base.market_writer import MarketWriter
from knowledge_base.wallet_writer import WalletWriter
from knowledge_base.execution_subscriber import ExecutionSubscriber
from knowledge_base.postmortem_generator import PostMortemGenerator

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("knowledge_base")


async def main() -> None:
    """Start the knowledge-base system."""
    _safe_url = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
    logger.info("knowledge-base (KBPM) starting  redis=%s", _safe_url)

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Init DB
    try:
        await init_db()
    except Exception:
        logger.warning("KBPM: database init failed -- running without Postgres", exc_info=True)

    # Task 1: Initialize vault
    vault_root = init_vault()

    # Initialize components
    market_writer = MarketWriter()
    pm_generator = PostMortemGenerator(redis_client, market_writer)

    thesis_writer = ThesisWriter(redis_client)
    wallet_writer = WalletWriter(redis_client)

    # ThesisWriter also needs to call MarketWriter
    original_thesis_process = thesis_writer._process

    async def thesis_process_with_market(thesis):
        await original_thesis_process(thesis)
        market_writer.write_from_thesis(thesis)

    thesis_writer._process = thesis_process_with_market

    execution_sub = ExecutionSubscriber(
        redis_client,
        market_writer,
        on_close=pm_generator.generate,
    )

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    components = [thesis_writer, wallet_writer, execution_sub]

    def _shutdown(sig_name: str) -> None:
        logger.info("knowledge-base received %s -- shutting down", sig_name)
        for comp in components:
            asyncio.ensure_future(comp.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
        except NotImplementedError:
            pass

    # Run all subscribers concurrently
    try:
        await asyncio.gather(
            thesis_writer.start(),
            wallet_writer.start(),
            execution_sub.start(),
        )
    finally:
        await redis_client.aclose()
        logger.info("knowledge-base (KBPM) stopped")


if __name__ == "__main__":
    asyncio.run(main())
