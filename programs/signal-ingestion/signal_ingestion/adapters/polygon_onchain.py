"""
Adapter 3: Polygon On-Chain Listener
Subscribes to Alchemy WebSocket for OrderFilled / OrderPlaced events on the
Polymarket CTF Exchange contract (Polygon mainnet).
Emits Signal(source_id=polygon_clob, category=on_chain) per event.
Publishes to redis channel: oracle:signal

Protocol:
  1. Connect to Alchemy Polygon WS endpoint (env: ALCHEMY_POLYGON_WS_URL).
  2. Subscribe to logs matching the CTF Exchange contract + event topics.
  3. Decode each log → Signal → Redis.
  4. On disconnect: reconnect with exponential back-off.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from web3 import AsyncWeb3
from web3.providers import WebSocketProvider

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

logger = logging.getLogger(__name__)

# Polymarket CTF Exchange on Polygon mainnet
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# OrderFilled(bytes32 orderHash, address maker, address taker, uint256 makerAssetId,
#             uint256 takerAssetId, uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)
ORDER_FILLED_TOPIC = AsyncWeb3.keccak(
    text="OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)"
).hex()

# OrderPlaced — not a standard CTF Exchange event; some CLOB wrappers emit it.
# We subscribe but silently ignore if absent.


class PolygonOnchainAdapter:
    """
    Shared adapter interface:
        start()  — connect, subscribe, stream (runs until stop())
        stop()   — signal the loop to exit
    """

    def __init__(
        self,
        redis_client: Any,
        ws_url: str | None = None,
        reconnect_delay: float = 5.0,
    ) -> None:
        self._redis = redis_client
        self._ws_url = ws_url or os.getenv(
            "ALCHEMY_POLYGON_WS_URL",
            "wss://polygon-mainnet.g.alchemy.com/v2/demo",
        )
        self._reconnect_delay = reconnect_delay
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        logger.info("PolygonOnchainAdapter started  ws=%s", self._ws_url[:60])
        delay = self._reconnect_delay
        while self._running:
            try:
                await self._connect_and_stream()
                delay = self._reconnect_delay
            except (ConnectionError, OSError, asyncio.TimeoutError) as exc:
                if not self._running:
                    break
                logger.warning(
                    "PolygonOnchainAdapter: connection error (%s) — retry in %gs",
                    exc, delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
            except Exception:
                if not self._running:
                    break
                logger.exception(
                    "PolygonOnchainAdapter: unexpected error — retry in %gs", delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def stop(self) -> None:
        self._running = False
        logger.info("PolygonOnchainAdapter stopped")

    # ── Core connect + stream ─────────────────────────────────────────────────

    async def _connect_and_stream(self) -> None:
        async with AsyncWeb3(WebSocketProvider(self._ws_url)) as w3:
            # Subscribe to OrderFilled logs on the CTF Exchange
            sub_id = await w3.eth.subscribe(
                "logs",
                {
                    "address": AsyncWeb3.to_checksum_address(CTF_EXCHANGE_ADDRESS),
                    "topics": [[ORDER_FILLED_TOPIC]],
                },
            )
            logger.info(
                "PolygonOnchainAdapter: subscribed  sub_id=%s", sub_id
            )

            async for msg in w3.socket.process_subscriptions():
                if not self._running:
                    break
                try:
                    log_entry = msg.get("result", msg)
                    signal = self._normalize(log_entry)
                    if signal is not None:
                        await self._publish(signal)
                except Exception:
                    logger.exception("PolygonOnchainAdapter: error handling log")

    # ── Normalize ─────────────────────────────────────────────────────────────

    def _normalize(self, log: Any) -> Signal | None:
        """
        Convert a raw Polygon log entry → Signal.

        raw_payload keys (per signal.md):
            tx_hash, wallet, market_id, outcome, side, price, size_usd,
            block_number, block_timestamp
        """
        tx_hash = log.get("transactionHash", "")
        if hasattr(tx_hash, "hex"):
            tx_hash = tx_hash.hex()

        block_number = log.get("blockNumber", 0)
        if isinstance(block_number, str):
            block_number = int(block_number, 16) if block_number.startswith("0x") else int(block_number)

        # Decode data field: 6 uint256 values packed (makerAssetId, takerAssetId,
        # makerAmountFilled, takerAmountFilled, fee — 5 words after the indexed topics)
        data = log.get("data", "0x")
        if hasattr(data, "hex"):
            data = "0x" + data.hex()
        topics = log.get("topics", [])

        maker = ""
        if len(topics) >= 3:
            raw_topic = topics[2]
            if hasattr(raw_topic, "hex"):
                raw_topic = raw_topic.hex()
            raw_topic = raw_topic if isinstance(raw_topic, str) else str(raw_topic)
            maker = "0x" + raw_topic[-40:]

        # Parse data words (each 32 bytes / 64 hex chars)
        data_hex = data[2:] if data.startswith("0x") else data
        words = [data_hex[i : i + 64] for i in range(0, len(data_hex), 64)]

        maker_asset_id = int(words[0], 16) if len(words) > 0 else 0
        taker_asset_id = int(words[1], 16) if len(words) > 1 else 0
        maker_filled = int(words[2], 16) if len(words) > 2 else 0
        taker_filled = int(words[3], 16) if len(words) > 3 else 0

        # Derive approximate price: taker_filled / maker_filled (USDC / shares)
        price = taker_filled / maker_filled if maker_filled > 0 else 0.0
        # Size in USDC (6 decimals)
        size_usd = taker_filled / 1e6

        # Determine side heuristic: if taker is paying USDC → buy
        side = "buy" if taker_filled > 0 else "sell"

        # market_id is the condition_id, which maps to the asset IDs on-chain.
        # We use the maker_asset_id as a proxy (consumer can resolve via token map).
        market_id = str(maker_asset_id)

        return Signal(
            source_id=SourceId.POLYGON_CLOB,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.ON_CHAIN,
            raw_payload={
                "tx_hash": tx_hash,
                "wallet": maker,
                "market_id": market_id,
                "outcome": str(taker_asset_id),
                "side": side,
                "price": round(price, 6),
                "size_usd": round(size_usd, 2),
                "block_number": block_number,
                "block_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            market_ids=[market_id] if market_id != "0" else None,
        )

    # ── Publish ───────────────────────────────────────────────────────────────

    async def _publish(self, signal: Signal) -> None:
        """Publish a normalized Signal to the canonical Redis channel."""
        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())
