"""Market scanner — sweeps all assets and ranks opportunities.

Scans crypto (CoinGecko) and stocks (Yahoo Finance) for pattern setups.
Publishes top opportunities to Redis for RE to analyze further.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from market_scanner.config import (
    CRYPTO_TOP_N,
    MIN_PATTERN_SCORE,
    SCANNER_SIGNAL_CHANNEL,
    SCANNER_STATE_KEY,
    STOCK_WATCHLIST,
)
from market_scanner.patterns import PatternResult, analyze
from market_scanner.providers.crypto import CryptoProvider
from market_scanner.providers.stocks import StockProvider

logger = logging.getLogger(__name__)


class MarketScanner:
    """Scan all markets for pattern-based trade setups."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._crypto = CryptoProvider()
        self._stocks = StockProvider()

    async def scan_all(self) -> list[PatternResult]:
        """Run a full scan of crypto + stocks. Returns sorted results."""
        results: list[PatternResult] = []

        # Scan crypto
        crypto_results = await self._scan_crypto()
        results.extend(crypto_results)

        # Scan stocks
        stock_results = await self._scan_stocks()
        results.extend(stock_results)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Filter by minimum score
        qualified = [r for r in results if r.score >= MIN_PATTERN_SCORE]

        # Publish top opportunities to Redis
        for result in qualified[:20]:  # top 20
            await self._publish(result)

        # Store scan summary in Redis
        summary = {
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "total_scanned": len(results),
            "qualified": len(qualified),
            "top_bull": [
                {"symbol": r.symbol, "score": r.score, "patterns": r.patterns}
                for r in qualified if r.direction.value == "bull"
            ][:5],
            "top_bear": [
                {"symbol": r.symbol, "score": r.score, "patterns": r.patterns}
                for r in qualified if r.direction.value == "bear"
            ][:5],
        }
        await self._redis.set(SCANNER_STATE_KEY, json.dumps(summary, default=str))

        logger.info(
            "MarketScanner: scanned %d assets  qualified=%d  "
            "bull=%d  bear=%d",
            len(results),
            len(qualified),
            sum(1 for r in qualified if r.direction.value == "bull"),
            sum(1 for r in qualified if r.direction.value == "bear"),
        )

        return qualified

    async def _scan_crypto(self) -> list[PatternResult]:
        """Scan top crypto assets."""
        results: list[PatternResult] = []
        try:
            assets = await self._crypto.get_top_assets(CRYPTO_TOP_N)
            logger.info("MarketScanner: scanning %d crypto assets", len(assets))

            for asset in assets:
                try:
                    df = await self._crypto.get_ohlcv(asset["id"], days=90)
                    result = analyze(df, asset["symbol"], "crypto", asset["id"])
                    if result:
                        results.append(result)
                except Exception:
                    logger.debug("MarketScanner: failed to scan crypto %s", asset["symbol"])
        except Exception:
            logger.warning("MarketScanner: crypto scan failed", exc_info=True)

        return results

    async def _scan_stocks(self) -> list[PatternResult]:
        """Scan stock watchlist."""
        results: list[PatternResult] = []
        tickers = [t.strip() for t in STOCK_WATCHLIST.split(",") if t.strip()]

        try:
            logger.info("MarketScanner: scanning %d stock tickers", len(tickers))
            for ticker in tickers:
                try:
                    df = await self._stocks.get_ohlcv(ticker, period="3mo")
                    result = analyze(df, ticker, "stock", ticker)
                    if result:
                        results.append(result)
                except Exception:
                    logger.debug("MarketScanner: failed to scan stock %s", ticker)
        except Exception:
            logger.warning("MarketScanner: stock scan failed", exc_info=True)

        return results

    async def _publish(self, result: PatternResult) -> None:
        """Publish a pattern result to Redis."""
        data = asdict(result)
        data["scanned_at"] = datetime.now(timezone.utc).isoformat()
        await self._redis.publish(SCANNER_SIGNAL_CHANNEL, json.dumps(data, default=str))
