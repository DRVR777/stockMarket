"""Binance WebSocket provider — real-time kline (candlestick) data.

Free, no API key required for public market data.
Supports any timeframe: 1m, 5m, 15m, 1h, 4h, 1d.

Usage::

    provider = BinanceKlineProvider()
    async for candle in provider.stream_klines(["BTCUSDT", "ETHUSDT"], interval="5m"):
        print(f"{candle.symbol} close={candle.close} volume={candle.volume}")
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator

import websockets

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"


@dataclass
class KlineCandle:
    """A single kline (candlestick) from Binance."""
    symbol: str
    interval: str
    open_time: int       # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int      # Unix ms
    is_closed: bool      # True when candle is finalized


class BinanceKlineProvider:
    """Stream real-time klines from Binance WebSocket (free, no auth)."""

    async def stream_klines(
        self,
        symbols: list[str],
        interval: str = "5m",
    ) -> AsyncIterator[KlineCandle]:
        """Stream kline candles for multiple symbols.

        ``symbols``: list of Binance pairs like ["BTCUSDT", "ETHUSDT"]
        ``interval``: "1m", "5m", "15m", "1h", "4h", "1d"

        Yields KlineCandle objects indefinitely.
        """
        # Build combined stream URL
        streams = "/".join(
            f"{s.lower()}@kline_{interval}" for s in symbols
        )
        url = f"{BINANCE_WS_BASE}/{streams}" if len(symbols) == 1 else \
              f"wss://stream.binance.com:9443/stream?streams={streams}"

        logger.info(
            "BinanceKlineProvider: connecting to %d symbols @ %s",
            len(symbols), interval,
        )

        async with websockets.connect(url, ping_interval=20) as ws:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    # Combined stream wraps in {"stream": "...", "data": {...}}
                    if "data" in data:
                        data = data["data"]

                    if data.get("e") != "kline":
                        continue

                    k = data["k"]
                    yield KlineCandle(
                        symbol=k["s"],
                        interval=k["i"],
                        open_time=k["t"],
                        open=float(k["o"]),
                        high=float(k["h"]),
                        low=float(k["l"]),
                        close=float(k["c"]),
                        volume=float(k["v"]),
                        close_time=k["T"],
                        is_closed=k["x"],
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug("BinanceKlineProvider: parse error: %s", e)
                    continue

    async def get_recent_klines(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 100,
    ) -> list[KlineCandle]:
        """Fetch recent historical klines via REST (for backfill).

        Uses the Binance public REST API — no auth needed.
        """
        import httpx

        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        candles = []
        for k in data:
            candles.append(KlineCandle(
                symbol=symbol,
                interval=interval,
                open_time=k[0],
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                close_time=k[6],
                is_closed=True,
            ))
        return candles
