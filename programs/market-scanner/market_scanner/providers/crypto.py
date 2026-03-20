"""Crypto data provider — CoinGecko API (free, no key required).

Fetches top N crypto assets by market cap with OHLCV history.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd
from pycoingecko import CoinGeckoAPI

logger = logging.getLogger(__name__)


class CryptoProvider:
    """Fetch crypto market data from CoinGecko."""

    def __init__(self) -> None:
        self._cg = CoinGeckoAPI()

    async def get_top_assets(self, n: int = 200) -> list[dict[str, Any]]:
        """Get top N crypto assets by market cap.

        Returns list of {id, symbol, name, market_cap, current_price}.
        """
        loop = asyncio.get_event_loop()
        pages = (n // 250) + 1
        all_coins: list[dict] = []

        for page in range(1, pages + 1):
            coins = await loop.run_in_executor(
                None,
                lambda p=page: self._cg.get_coins_markets(
                    vs_currency="usd",
                    order="market_cap_desc",
                    per_page=min(250, n),
                    page=p,
                ),
            )
            all_coins.extend(coins)
            if len(all_coins) >= n:
                break

        return [
            {
                "id": c["id"],
                "symbol": c["symbol"].upper(),
                "name": c["name"],
                "market_cap": c.get("market_cap", 0),
                "current_price": c.get("current_price", 0),
                "asset_type": "crypto",
            }
            for c in all_coins[:n]
        ]

    async def get_ohlcv(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """Get OHLCV data for a single crypto asset.

        CoinGecko free tier returns daily prices. Returns DataFrame with
        columns: [open, high, low, close, volume].
        """
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: self._cg.get_coin_ohlc_by_id(
                id=coin_id,
                vs_currency="usd",
                days=str(days),
            ),
        )

        if not data:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # CoinGecko OHLC doesn't include volume — fetch separately
        market_data = await loop.run_in_executor(
            None,
            lambda: self._cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency="usd",
                days=str(days),
            ),
        )
        volumes = market_data.get("total_volumes", [])
        if volumes:
            vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
            vol_df.set_index("timestamp", inplace=True)
            vol_df = vol_df.resample("D").sum()
            df = df.resample("D").last()
            df["volume"] = vol_df["volume"].reindex(df.index, method="nearest").fillna(0)
        else:
            df["volume"] = 0.0

        return df.dropna()
