"""Live scanner — real-time market scanning with ML filtering.

Combines all scanning layers:
  1. Binance WebSocket for real-time crypto klines
  2. SMC pattern detection on each completed candle
  3. ML model filters to high-probability setups only
  4. Web scraper context (fear/greed, funding rates, Reddit sentiment)
  5. Publishes qualified signals to oracle:signal for RE pipeline

This is the main entry point for continuous market monitoring.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId

from market_scanner.smc import analyze_smc, Bias, SMCAnalysis
from market_scanner.ml_classifier import FEATURE_NAMES, extract_features
from market_scanner.providers.binance_ws import BinanceKlineProvider, KlineCandle
from market_scanner.providers.scrapers import (
    scrape_fear_greed,
    scrape_funding_rates,
    scrape_reddit_trending,
)
from market_scanner.config import MIN_PATTERN_SCORE, SCANNER_SIGNAL_CHANNEL

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """A qualified trade setup from the live scanner."""
    symbol: str
    direction: str      # "long" | "short"
    confidence: float   # SMC confidence
    ml_probability: float  # ML win probability
    setup_type: str
    signals: list[str]
    entry_zone: tuple[float, float] | None
    stop_loss: float | None
    take_profit: float | None
    current_price: float
    fear_greed: int
    funding_rate: float


class LiveScanner:
    """Continuous market scanner with ML model gate."""

    def __init__(
        self,
        redis_client: Any,
        symbols: list[str] | None = None,
        interval: str = "15m",
        ml_threshold: float = 0.55,
    ) -> None:
        self._redis = redis_client
        self._symbols = symbols or [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "APTUSDT",
        ]
        self._interval = interval
        self._ml_threshold = ml_threshold
        self._binance = BinanceKlineProvider()
        self._ml_model = None
        self._running = False

        # Per-symbol candle buffer for building DataFrames
        self._candle_buffers: dict[str, list[dict]] = defaultdict(list)
        self._max_buffer = 200  # keep last 200 candles per symbol

        # Market context (refreshed periodically)
        self._fear_greed = 50
        self._funding_rates: dict[str, float] = {}

    async def start(self) -> None:
        """Start the live scanner."""
        self._running = True

        # Load ML model
        self._load_model()

        # Backfill candle buffers
        await self._backfill()

        # Refresh market context
        await self._refresh_context()

        # Start context refresh loop in background
        asyncio.create_task(self._context_refresh_loop())

        # Stream candles and scan
        logger.info(
            "LiveScanner: streaming %d symbols @ %s  ml_threshold=%.0f%%",
            len(self._symbols), self._interval, self._ml_threshold * 100,
        )

        async for candle in self._binance.stream_klines(self._symbols, self._interval):
            if not self._running:
                break

            # Update buffer
            self._update_buffer(candle)

            # Only analyze on closed candles (complete bars)
            if not candle.is_closed:
                continue

            # Run analysis
            result = self._analyze(candle.symbol)
            if result and result.ml_probability >= self._ml_threshold:
                await self._publish(result)
                logger.info(
                    "SIGNAL: %s %s  smc=%.2f  ml=%.1f%%  setup=%s  price=$%.2f",
                    result.symbol, result.direction, result.confidence,
                    result.ml_probability * 100, result.setup_type, result.current_price,
                )

    async def stop(self) -> None:
        self._running = False
        logger.info("LiveScanner: stopped")

    def _load_model(self) -> None:
        """Load the best available ML model (newest/best first)."""
        base = Path(__file__).parent.parent  # market-scanner root
        model_paths = [
            # V3 first (best: 108k trades, 68% accuracy)
            base / "models" / "v3_latest.pkl",
            base / "models" / "v3_108368trades_68pct.pkl",
            Path("models") / "v3_latest.pkl",
            # V2 fallback
            base / "models" / "v2_expanded.pkl",
            Path("models") / "v2_expanded.pkl",
            # V1 fallback
            base / "models" / "expanded_classifier.pkl",
            Path("models") / "expanded_classifier.pkl",
        ]
        for path in model_paths:
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, dict):
                        self._ml_model = data.get("model")
                        self._ml_features = data.get("features", FEATURE_NAMES)
                    else:
                        self._ml_model = data
                        self._ml_features = FEATURE_NAMES
                    logger.info("LiveScanner: loaded ML model from %s", path.name)
                    return
                except Exception:
                    continue
        logger.warning("LiveScanner: no ML model found — running without ML filter")

    async def _backfill(self) -> None:
        """Fetch historical klines to fill the candle buffers."""
        logger.info("LiveScanner: backfilling %d symbols...", len(self._symbols))
        for sym in self._symbols:
            try:
                candles = await self._binance.get_recent_klines(
                    sym, interval=self._interval, limit=self._max_buffer,
                )
                for c in candles:
                    self._candle_buffers[sym].append({
                        "open": c.open, "high": c.high, "low": c.low,
                        "close": c.close, "volume": c.volume,
                    })
                logger.debug("  %s: %d candles", sym, len(candles))
            except Exception:
                logger.warning("  %s: backfill failed", sym)

    def _update_buffer(self, candle: KlineCandle) -> None:
        """Add a candle to the buffer."""
        buf = self._candle_buffers[candle.symbol]
        entry = {
            "open": candle.open, "high": candle.high,
            "low": candle.low, "close": candle.close,
            "volume": candle.volume,
        }
        if candle.is_closed:
            buf.append(entry)
            if len(buf) > self._max_buffer:
                buf.pop(0)
        elif buf:
            # Update the last (current) candle
            buf[-1] = entry

    def _analyze(self, symbol: str) -> ScanResult | None:
        """Run SMC + ML analysis on a symbol's candle buffer."""
        buf = self._candle_buffers.get(symbol)
        if not buf or len(buf) < 60:
            return None

        df = pd.DataFrame(buf)

        # SMC analysis
        smc = analyze_smc(df, symbol, "crypto")
        if smc is None or smc.bias == Bias.NEUTRAL:
            return None
        if smc.confidence < MIN_PATTERN_SCORE:
            return None

        # ML prediction
        ml_prob = 0.5
        if self._ml_model is not None:
            try:
                direction = "long" if smc.bias == Bias.BULLISH else "short"
                feats = extract_features(df, len(df) - 1, direction)
                feats["bias_is_bullish"] = 1.0 if direction == "long" else 0.0
                # Add v2 features if available
                feats["funding_rate"] = self._funding_rates.get(symbol, 0) * 10000
                feats["fear_greed"] = float(self._fear_greed)

                import numpy as np
                X = np.array([[feats.get(name, 0.0) for name in self._ml_features]])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                proba = self._ml_model.predict_proba(X)[0]
                ml_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception:
                logger.debug("LiveScanner: ML predict failed for %s", symbol, exc_info=True)

        direction = "long" if smc.bias == Bias.BULLISH else "short"

        return ScanResult(
            symbol=symbol,
            direction=direction,
            confidence=smc.confidence,
            ml_probability=ml_prob,
            setup_type=smc.setup_type or "smc_generic",
            signals=smc.signals[:5],
            entry_zone=smc.entry_zone,
            stop_loss=smc.stop_loss,
            take_profit=smc.take_profit,
            current_price=smc.current_price,
            fear_greed=self._fear_greed,
            funding_rate=self._funding_rates.get(symbol, 0),
        )

    async def _publish(self, result: ScanResult) -> None:
        """Publish a qualified setup to Redis."""
        signal = Signal(
            source_id=SourceId.AI_OPINION,
            timestamp=datetime.now(timezone.utc),
            category=SignalCategory.PRICE,
            raw_payload={
                "scanner": "live_smc_ml",
                "symbol": result.symbol,
                "direction": result.direction,
                "smc_confidence": result.confidence,
                "ml_win_probability": result.ml_probability,
                "setup_type": result.setup_type,
                "signals": result.signals,
                "entry_zone": list(result.entry_zone) if result.entry_zone else None,
                "stop_loss": result.stop_loss,
                "take_profit": result.take_profit,
                "current_price": result.current_price,
                "fear_greed": result.fear_greed,
                "funding_rate": result.funding_rate,
            },
            confidence=result.ml_probability,
        )
        await self._redis.publish(Signal.CHANNEL, signal.model_dump_json())
        await self._redis.publish(SCANNER_SIGNAL_CHANNEL, json.dumps({
            "symbol": result.symbol,
            "direction": result.direction,
            "smc_confidence": result.confidence,
            "ml_probability": result.ml_probability,
            "setup_type": result.setup_type,
            "price": result.current_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    async def _refresh_context(self) -> None:
        """Refresh market-wide context data."""
        try:
            fg = await scrape_fear_greed()
            if fg:
                self._fear_greed = fg.value
        except Exception:
            pass

        try:
            rates = await scrape_funding_rates()
            self._funding_rates = {r.symbol: r.rate for r in rates}
        except Exception:
            pass

        logger.debug(
            "LiveScanner: context refreshed  fear_greed=%d  funding_pairs=%d",
            self._fear_greed, len(self._funding_rates),
        )

    async def _context_refresh_loop(self) -> None:
        """Refresh market context every 5 minutes."""
        while self._running:
            await asyncio.sleep(300)
            await self._refresh_context()
