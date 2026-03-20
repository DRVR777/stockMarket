"""solana-executor entry point.

Usage:  uv run python -m solana_executor
        (or via Makefile: make solana-executor)

Starts in paper trading mode by default (SOE_PAPER_TRADING=true).
Set SOE_PAPER_TRADING=false after checkpoint approval for live execution.

The execution engine is chain-agnostic. The Solana adapter is loaded by
default but can be swapped for any ChainAdapter implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

import redis.asyncio as aioredis
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from oracle_shared.db import init_db

from solana_executor.config import (
    FLOOR_ESTIMATE_INTERVAL_HOURS,
    LOG_FORMAT,
    LOG_LEVEL,
    PAPER_TRADING,
    PARAMS_KEY,
    REDIS_URL,
)
from solana_executor.chains.solana import SolanaAdapter
from solana_executor.statistical_model import AssetModel, ModelStore
from solana_executor.entry_exit import EntryExitEngine
from solana_executor.floor_requester import FloorRequester

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("solana_executor")


async def main() -> None:
    """Start the SOE mean-reversion trading engine."""
    logger.info(
        "solana-executor (SOE) starting  redis=%s  paper=%s",
        REDIS_URL, PAPER_TRADING,
    )

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Init DB
    try:
        await init_db()
    except Exception:
        logger.warning("SOE: database init failed — running without Postgres", exc_info=True)

    # ── Task 1: Load configured assets ────────────────────────────────────────
    chain = SolanaAdapter()
    model_store = ModelStore(redis_client)
    entry_exit = EntryExitEngine(redis_client, chain)
    floor_req = FloorRequester(redis_client, model_store)

    assets = await _load_assets(redis_client)
    if not assets:
        logger.warning("SOE: no assets configured at oracle:state:params:soe_assets — using defaults")
        assets = [
            {"token_address": "So11111111111111111111111111111111111111112", "symbol": "SOL"},
            {"token_address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN", "symbol": "JUP"},
        ]

    # ── Task 2: Backfill OHLCV ────────────────────────────────────────────────
    models: dict[str, AssetModel] = {}
    for asset in assets:
        addr = asset["token_address"]
        symbol = asset["symbol"]
        model = await model_store.load(addr)
        if model is None:
            model = AssetModel(
                token_address=addr,
                symbol=symbol,
                chain=chain.chain_name,
            )

        try:
            bars = await chain.get_ohlcv(addr, days=30)
            model.update_from_ohlcv(bars)
            await model_store.save(model)
            logger.info("SOE: backfilled %s — %d bars  ma20=$%.4f", symbol, len(bars), model.ma_20)
        except Exception:
            logger.warning("SOE: OHLCV backfill failed for %s", symbol, exc_info=True)

        models[addr] = model

    # ── Task 4: Schedule floor estimate requests ──────────────────────────────
    scheduler = AsyncIOScheduler()

    async def request_all_floors() -> None:
        for addr, model in models.items():
            await floor_req.request_floor(model)

    scheduler.add_job(
        request_all_floors,
        "interval",
        hours=FLOOR_ESTIMATE_INTERVAL_HOURS,
        id="floor_estimates",
        replace_existing=True,
    )
    scheduler.start()

    # Request floors once at startup
    asyncio.ensure_future(request_all_floors())

    # ── Task 3: Subscribe to price feed + entry/exit loop ─────────────────────
    running = True

    def _shutdown(sig_name: str) -> None:
        nonlocal running
        logger.info("SOE received %s — shutting down", sig_name)
        running = False

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
        except NotImplementedError:
            pass

    try:
        token_addresses = [a["token_address"] for a in assets]
        logger.info("SOE: subscribing to %d assets via %s", len(assets), chain.chain_name)

        async for tick in chain.subscribe_prices(token_addresses):
            if not running:
                break

            addr = tick.token_address
            if addr in models:
                models[addr].update_price(tick.price_usd)
                await model_store.save_price(addr, tick.price_usd)

                # Task 5: Check entry
                await entry_exit.check_entry(models[addr])

            # Task 7: Check exits on every tick
            current_prices = {a: m.current_price for a, m in models.items()}
            await entry_exit.check_exits(current_prices)

    except (ConnectionError, OSError) as e:
        if running:
            logger.error("SOE: connection error: %s", e)
    finally:
        scheduler.shutdown(wait=False)
        await chain.close()
        await redis_client.aclose()
        logger.info("solana-executor (SOE) stopped")


async def _load_assets(redis_client: aioredis.Redis) -> list[dict]:
    """Load configured assets from Redis params."""
    raw = await redis_client.hget(PARAMS_KEY, "soe_assets")
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return []


if __name__ == "__main__":
    asyncio.run(main())
