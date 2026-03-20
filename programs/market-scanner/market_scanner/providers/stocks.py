"""Stock data provider — Yahoo Finance (free, no key required).

Fetches OHLCV for any US stock/ETF ticker.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class StockProvider:
    """Fetch stock market data from Yahoo Finance."""

    async def get_watchlist(self, tickers: list[str]) -> list[dict[str, Any]]:
        """Build asset info for a list of ticker symbols."""
        loop = asyncio.get_event_loop()
        assets = []

        for ticker in tickers:
            try:
                info = await loop.run_in_executor(
                    None,
                    lambda t=ticker: yf.Ticker(t).fast_info,
                )
                assets.append({
                    "id": ticker,
                    "symbol": ticker,
                    "name": ticker,
                    "market_cap": getattr(info, "market_cap", 0) or 0,
                    "current_price": getattr(info, "last_price", 0) or 0,
                    "asset_type": "stock",
                })
            except Exception:
                logger.warning("StockProvider: failed to fetch info for %s", ticker)
                assets.append({
                    "id": ticker,
                    "symbol": ticker,
                    "name": ticker,
                    "market_cap": 0,
                    "current_price": 0,
                    "asset_type": "stock",
                })

        return assets

    async def get_ohlcv(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
        """Get OHLCV data for a stock ticker.

        Returns DataFrame with columns: [open, high, low, close, volume].
        """
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: yf.download(ticker, period=period, progress=False, auto_adjust=True),
        )

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
