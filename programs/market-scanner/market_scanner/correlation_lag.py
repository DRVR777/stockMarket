"""Correlation & lag detector — find which assets lead and which follow.

BTC moves first, alts follow 5-15 minutes later. This module:
1. Streams real-time prices from Binance
2. Computes rolling correlation between asset pairs
3. Detects lag (which asset moves first, by how many bars)
4. When BTC makes a sharp move, predicts which alts will follow and when
5. Emits actionable signals: "SOL likely to drop 1.2% in next 10 min"

Usage::

    detector = LagDetector(redis_client)
    await detector.start()  # streams and emits signals
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from market_scanner.providers.binance_ws import BinanceKlineProvider, KlineCandle

logger = logging.getLogger(__name__)

# Assets to track — BTC is the leader, rest are followers
LEADER = "BTCUSDT"
FOLLOWERS = [
    "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "NEARUSDT", "MATICUSDT", "UNIUSDT",
]

# Detection thresholds
SHARP_MOVE_PCT = 0.3       # BTC moves 0.3%+ in one bar = sharp move
LOOKBACK_BARS = 60         # rolling window for correlation
LAG_SEARCH_BARS = 15       # max lag to search (15 bars @ 1min = 15 min)
MIN_CORRELATION = 0.5      # minimum correlation to consider a pair
SIGNAL_CHANNEL = "oracle:lag_signal"


@dataclass
class LagProfile:
    """Measured lag relationship between leader and follower."""
    follower: str
    correlation: float      # -1 to +1, how closely they move together
    lag_bars: int           # how many bars the follower lags behind leader
    avg_follow_pct: float   # average % move of follower per 1% leader move (beta)
    confidence: int         # how many observations this is based on
    last_updated: float


@dataclass
class LagSignal:
    """Actionable signal: follower expected to move soon."""
    leader: str
    follower: str
    leader_move_pct: float     # BTC moved this much
    expected_move_pct: float   # follower expected to move this much
    expected_direction: str    # "up" or "down"
    expected_lag_minutes: int  # follower expected to move in N minutes
    confidence: float          # 0-1
    timestamp: str


class LagDetector:
    """Real-time cross-asset lag detection and signal generation."""

    def __init__(
        self,
        redis_client: Any | None = None,
        interval: str = "1m",
    ) -> None:
        self._redis = redis_client
        self._interval = interval
        self._binance = BinanceKlineProvider()
        self._running = False

        # Price buffers: symbol -> deque of close prices
        self._prices: dict[str, deque] = defaultdict(lambda: deque(maxlen=LOOKBACK_BARS))
        # Return buffers: symbol -> deque of % returns
        self._returns: dict[str, deque] = defaultdict(lambda: deque(maxlen=LOOKBACK_BARS))
        # Last known price per symbol
        self._last_price: dict[str, float] = {}

        # Computed lag profiles
        self._profiles: dict[str, LagProfile] = {}

        # Recent leader moves for signal generation
        self._leader_moves: deque = deque(maxlen=20)

    async def start(self) -> None:
        """Stream prices and detect lag signals."""
        self._running = True
        all_symbols = [LEADER] + FOLLOWERS

        # Backfill price buffers
        await self._backfill(all_symbols)

        # Compute initial lag profiles
        self._recompute_profiles()

        logger.info(
            "LagDetector: streaming %d symbols @ %s  leader=%s",
            len(all_symbols), self._interval, LEADER,
        )

        bar_count = 0
        async for candle in self._binance.stream_klines(all_symbols, self._interval):
            if not self._running:
                break

            # Update price buffer on closed candles
            if candle.is_closed:
                self._update(candle)
                bar_count += 1

                # Recompute profiles every 30 bars
                if bar_count % 30 == 0:
                    self._recompute_profiles()

                # Check for sharp leader moves
                if candle.symbol == LEADER:
                    await self._check_leader_move()

    async def stop(self) -> None:
        self._running = False
        logger.info("LagDetector: stopped")

    async def _backfill(self, symbols: list[str]) -> None:
        """Fill price buffers with recent data."""
        for sym in symbols:
            try:
                candles = await self._binance.get_recent_klines(
                    sym, interval=self._interval, limit=LOOKBACK_BARS,
                )
                for c in candles:
                    self._prices[sym].append(c.close)
                    self._last_price[sym] = c.close

                # Compute returns
                prices = list(self._prices[sym])
                for i in range(1, len(prices)):
                    ret = (prices[i] - prices[i-1]) / prices[i-1] * 100
                    self._returns[sym].append(ret)
            except Exception:
                pass
        logger.info("LagDetector: backfilled %d symbols", len(symbols))

    def _update(self, candle: KlineCandle) -> None:
        """Update buffers with new candle data."""
        sym = candle.symbol
        prev = self._last_price.get(sym, candle.close)
        self._prices[sym].append(candle.close)
        self._last_price[sym] = candle.close

        ret = (candle.close - prev) / prev * 100 if prev > 0 else 0
        self._returns[sym].append(ret)

    def _recompute_profiles(self) -> None:
        """Compute lag profiles for all followers vs leader."""
        leader_returns = list(self._returns[LEADER])
        if len(leader_returns) < 20:
            return

        leader_arr = np.array(leader_returns)

        for follower in FOLLOWERS:
            follower_returns = list(self._returns[follower])
            if len(follower_returns) < 20:
                continue

            follower_arr = np.array(follower_returns)
            min_len = min(len(leader_arr), len(follower_arr))

            # Find optimal lag by cross-correlation
            best_corr = 0
            best_lag = 0

            for lag in range(0, min(LAG_SEARCH_BARS, min_len - 10)):
                if lag == 0:
                    l = leader_arr[-min_len:]
                    f = follower_arr[-min_len:]
                else:
                    # Leader leads by `lag` bars: compare leader[:-lag] with follower[lag:]
                    l = leader_arr[-(min_len - lag):-lag] if lag > 0 else leader_arr[-min_len:]
                    f = follower_arr[-min_len + lag:]

                if len(l) < 10 or len(f) < 10:
                    continue
                actual_len = min(len(l), len(f))
                l = l[:actual_len]
                f = f[:actual_len]

                if np.std(l) == 0 or np.std(f) == 0:
                    continue

                corr = float(np.corrcoef(l, f)[0, 1])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            # Compute beta (average follower move per 1% leader move)
            if best_lag == 0:
                l_slice = leader_arr[-min_len:]
                f_slice = follower_arr[-min_len:]
            else:
                l_slice = leader_arr[-(min_len - best_lag):-best_lag]
                f_slice = follower_arr[-min_len + best_lag:]
            actual = min(len(l_slice), len(f_slice))
            l_slice = l_slice[:actual]
            f_slice = f_slice[:actual]

            # Beta = slope of regression
            if np.std(l_slice) > 0:
                beta = float(np.cov(l_slice, f_slice)[0, 1] / np.var(l_slice))
            else:
                beta = 1.0

            self._profiles[follower] = LagProfile(
                follower=follower,
                correlation=round(best_corr, 3),
                lag_bars=best_lag,
                avg_follow_pct=round(beta, 3),
                confidence=min_len,
                last_updated=time.time(),
            )

    async def _check_leader_move(self) -> None:
        """Check if leader just made a sharp move. If so, emit signals for followers."""
        returns = list(self._returns[LEADER])
        if not returns:
            return

        last_return = returns[-1]
        if abs(last_return) < SHARP_MOVE_PCT:
            return

        # Sharp move detected!
        direction = "down" if last_return < 0 else "up"
        logger.info(
            "LagDetector: SHARP MOVE %s %s %.2f%%",
            LEADER, direction, last_return,
        )

        # Emit signals for followers
        for follower, profile in self._profiles.items():
            if abs(profile.correlation) < MIN_CORRELATION:
                continue

            expected_move = last_return * profile.avg_follow_pct
            # If negative correlation, follower moves opposite
            if profile.correlation < 0:
                expected_move = -expected_move

            expected_dir = "down" if expected_move < 0 else "up"
            lag_minutes = profile.lag_bars  # at 1m interval, lag_bars = minutes

            signal = LagSignal(
                leader=LEADER,
                follower=follower,
                leader_move_pct=round(last_return, 3),
                expected_move_pct=round(expected_move, 3),
                expected_direction=expected_dir,
                expected_lag_minutes=lag_minutes,
                confidence=min(1.0, abs(profile.correlation) * (profile.confidence / LOOKBACK_BARS)),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(
                "  -> %s expected %s %.2f%% in %d min (corr=%.2f beta=%.2f)",
                follower, expected_dir, abs(expected_move),
                lag_minutes, profile.correlation, profile.avg_follow_pct,
            )

            if self._redis:
                await self._redis.publish(
                    SIGNAL_CHANNEL,
                    json.dumps({
                        "leader": signal.leader,
                        "follower": signal.follower,
                        "leader_move_pct": signal.leader_move_pct,
                        "expected_move_pct": signal.expected_move_pct,
                        "expected_direction": signal.expected_direction,
                        "expected_lag_minutes": signal.expected_lag_minutes,
                        "confidence": signal.confidence,
                        "timestamp": signal.timestamp,
                    }),
                )

    def get_profiles(self) -> dict[str, LagProfile]:
        """Return current lag profiles."""
        return dict(self._profiles)

    def get_rankings(self) -> list[tuple[str, float, int, float]]:
        """Return followers ranked by correlation strength.

        Returns list of (symbol, correlation, lag_bars, beta).
        """
        ranked = sorted(
            self._profiles.values(),
            key=lambda p: abs(p.correlation),
            reverse=True,
        )
        return [
            (p.follower, p.correlation, p.lag_bars, p.avg_follow_pct)
            for p in ranked
        ]
