"""market-scanner entry point.

Usage:  python -m market_scanner
        (or via Makefile: make market-scanner)

Scans crypto (CoinGecko) and stocks (Yahoo Finance) for technical patterns.
Publishes qualified setups to oracle:scanner_signal for RE analysis.
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

from market_scanner.config import (
    CRYPTO_SCAN_INTERVAL,
    LOG_FORMAT,
    LOG_LEVEL,
    REDIS_URL,
    STOCK_SCAN_INTERVAL,
)
from market_scanner.scanner import MarketScanner

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("market_scanner")


async def main() -> None:
    logger.info("market-scanner starting  redis=%s", REDIS_URL)

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    scanner = MarketScanner(redis_client)

    running = True

    def _shutdown(sig_name: str) -> None:
        nonlocal running
        logger.info("market-scanner received %s -- shutting down", sig_name)
        running = False

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
        except NotImplementedError:
            pass

    try:
        while running:
            logger.info("Starting full market scan...")
            try:
                results = await scanner.scan_all()

                # Print top opportunities
                for r in results[:10]:
                    direction = "BULL" if r.direction.value == "bull" else "BEAR" if r.direction.value == "bear" else "---"
                    logger.info(
                        "  [%s] %s %-6s  score=%.2f  rsi=%.0f  patterns=%s  "
                        "entry=$%.2f  sl=$%.2f  tp=$%.2f",
                        direction,
                        r.asset_type[:5],
                        r.symbol,
                        r.score,
                        r.rsi,
                        ",".join(r.patterns[:3]),
                        r.entry_price or 0,
                        r.stop_loss or 0,
                        r.take_profit or 0,
                    )
            except Exception:
                logger.exception("market-scanner: scan cycle failed")

            # Wait for next cycle
            for _ in range(min(CRYPTO_SCAN_INTERVAL, STOCK_SCAN_INTERVAL)):
                if not running:
                    break
                await asyncio.sleep(1)
    finally:
        await redis_client.aclose()
        logger.info("market-scanner stopped")


if __name__ == "__main__":
    asyncio.run(main())
