"""Pattern detection engine — technical analysis for bull/bear setups.

Scans OHLCV data for:
  - Momentum: RSI reversals, MACD crossovers
  - Trend: MA crossovers (golden/death cross), higher-high/lower-low structure
  - Volatility: Bollinger Band squeeze → expansion
  - Volume: unusual volume spikes
  - Support/Resistance: breakouts and breakdowns

Returns a PatternResult with a composite score (0-1) and detected patterns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import ta

from market_scanner.config import (
    BB_SQUEEZE_PERCENTILE,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    VOLUME_SPIKE_MULTIPLIER,
)

logger = logging.getLogger(__name__)


class Direction(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


@dataclass
class PatternResult:
    """Result of pattern analysis on a single asset."""
    symbol: str
    asset_type: str  # "crypto" | "stock"
    asset_id: str
    direction: Direction
    score: float  # 0.0-1.0 composite pattern strength
    patterns: list[str]  # detected pattern names
    current_price: float
    rsi: float
    macd_signal: str  # "bullish_cross" | "bearish_cross" | "neutral"
    ma_trend: str  # "above_ma" | "below_ma" | "golden_cross" | "death_cross"
    volume_ratio: float  # current volume / 20-day average
    bb_squeeze: bool
    support: float
    resistance: float
    entry_price: Optional[float] = None  # suggested entry
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


def analyze(df: pd.DataFrame, symbol: str, asset_type: str, asset_id: str) -> Optional[PatternResult]:
    """Run full pattern analysis on OHLCV data.

    Requires at least 30 rows of daily data.
    Returns None if data is insufficient.
    """
    if len(df) < 26:  # need at least 26 rows for MACD
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    current_price = float(close.iloc[-1])

    patterns: list[str] = []
    bull_score = 0.0
    bear_score = 0.0

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) > 1 else rsi

    if rsi < RSI_OVERSOLD:
        patterns.append("rsi_oversold")
        bull_score += 0.2
    elif rsi > RSI_OVERBOUGHT:
        patterns.append("rsi_overbought")
        bear_score += 0.2

    # RSI reversal from oversold
    if rsi_prev < RSI_OVERSOLD and rsi > RSI_OVERSOLD:
        patterns.append("rsi_bullish_reversal")
        bull_score += 0.15
    # RSI rejection from overbought
    if rsi_prev > RSI_OVERBOUGHT and rsi < RSI_OVERBOUGHT:
        patterns.append("rsi_bearish_reversal")
        bear_score += 0.15

    # ── MACD ───────────────────────────────────────────────────────────────
    macd_ind = ta.trend.MACD(close)
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()
    macd_signal = "neutral"

    if len(macd_line) >= 2 and len(signal_line) >= 2:
        macd_now = float(macd_line.iloc[-1])
        macd_prev = float(macd_line.iloc[-2])
        sig_now = float(signal_line.iloc[-1])
        sig_prev = float(signal_line.iloc[-2])

        # Bullish crossover: MACD crosses above signal
        if macd_prev <= sig_prev and macd_now > sig_now:
            macd_signal = "bullish_cross"
            patterns.append("macd_bullish_cross")
            bull_score += 0.2
        # Bearish crossover
        elif macd_prev >= sig_prev and macd_now < sig_now:
            macd_signal = "bearish_cross"
            patterns.append("macd_bearish_cross")
            bear_score += 0.2

    # ── Moving Averages ────────────────────────────────────────────────────
    ma_20 = float(close.rolling(20).mean().iloc[-1])
    ma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma_20
    ma_trend = "neutral"

    if current_price > ma_20:
        ma_trend = "above_ma"
        bull_score += 0.1
    else:
        ma_trend = "below_ma"
        bear_score += 0.1

    # Golden cross (MA20 crosses above MA50)
    if len(close) >= 50:
        ma20_series = close.rolling(20).mean()
        ma50_series = close.rolling(50).mean()
        if (float(ma20_series.iloc[-2]) <= float(ma50_series.iloc[-2])
                and float(ma20_series.iloc[-1]) > float(ma50_series.iloc[-1])):
            ma_trend = "golden_cross"
            patterns.append("golden_cross")
            bull_score += 0.25
        elif (float(ma20_series.iloc[-2]) >= float(ma50_series.iloc[-2])
                and float(ma20_series.iloc[-1]) < float(ma50_series.iloc[-1])):
            ma_trend = "death_cross"
            patterns.append("death_cross")
            bear_score += 0.25

    # ── Volume ─────────────────────────────────────────────────────────────
    vol_avg = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
    vol_now = float(volume.iloc[-1])
    volume_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0

    if volume_ratio >= VOLUME_SPIKE_MULTIPLIER:
        if current_price > float(close.iloc[-2]):
            patterns.append("volume_breakout_up")
            bull_score += 0.15
        else:
            patterns.append("volume_breakout_down")
            bear_score += 0.15

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_bandwidth = bb.bollinger_wband()
    bb_squeeze = False

    if len(bb_bandwidth) >= 20:
        current_bw = float(bb_bandwidth.iloc[-1])
        historical_bw = bb_bandwidth.iloc[-60:] if len(bb_bandwidth) >= 60 else bb_bandwidth
        percentile = float((historical_bw < current_bw).mean())
        if percentile <= BB_SQUEEZE_PERCENTILE:
            bb_squeeze = True
            patterns.append("bb_squeeze")
            # Squeeze is directionally neutral — amplifies other signals
            bull_score *= 1.2
            bear_score *= 1.2

    # ── Support / Resistance ───────────────────────────────────────────────
    recent_lows = low.iloc[-20:]
    recent_highs = high.iloc[-20:]
    support = float(recent_lows.min())
    resistance = float(recent_highs.max())

    # Breakout above resistance
    if current_price > resistance * 0.99 and current_price < resistance * 1.02:
        patterns.append("resistance_test")
        bull_score += 0.1
    # Breakdown below support
    if current_price < support * 1.01 and current_price > support * 0.98:
        patterns.append("support_test")
        bear_score += 0.1

    # ── Higher-High / Lower-Low Structure ──────────────────────────────────
    if len(high) >= 10:
        recent_h = high.iloc[-10:]
        recent_l = low.iloc[-10:]
        mid = len(recent_h) // 2
        if float(recent_h.iloc[mid:].max()) > float(recent_h.iloc[:mid].max()):
            if float(recent_l.iloc[mid:].min()) > float(recent_l.iloc[:mid].min()):
                patterns.append("higher_high_higher_low")
                bull_score += 0.15
        if float(recent_h.iloc[mid:].max()) < float(recent_h.iloc[:mid].max()):
            if float(recent_l.iloc[mid:].min()) < float(recent_l.iloc[:mid].min()):
                patterns.append("lower_high_lower_low")
                bear_score += 0.15

    # ── Composite scoring ──────────────────────────────────────────────────
    bull_score = min(1.0, bull_score)
    bear_score = min(1.0, bear_score)

    if bull_score > bear_score:
        direction = Direction.BULL
        score = bull_score
        entry_price = current_price
        stop_loss = support * 0.98
        take_profit = current_price * 1.08
    elif bear_score > bull_score:
        direction = Direction.BEAR
        score = bear_score
        entry_price = current_price
        stop_loss = resistance * 1.02
        take_profit = current_price * 0.92
    else:
        direction = Direction.NEUTRAL
        score = 0.0
        entry_price = None
        stop_loss = None
        take_profit = None

    return PatternResult(
        symbol=symbol,
        asset_type=asset_type,
        asset_id=asset_id,
        direction=direction,
        score=round(score, 3),
        patterns=patterns,
        current_price=current_price,
        rsi=round(rsi, 1),
        macd_signal=macd_signal,
        ma_trend=ma_trend,
        volume_ratio=round(volume_ratio, 2),
        bb_squeeze=bb_squeeze,
        support=round(support, 4),
        resistance=round(resistance, 4),
        entry_price=round(entry_price, 4) if entry_price else None,
        stop_loss=round(stop_loss, 4) if stop_loss else None,
        take_profit=round(take_profit, 4) if take_profit else None,
    )
