"""Market regime detector — classify current market conditions.

Markets cycle through regimes:
  - TRENDING UP: strong uptrend, buy dips, trail stops
  - TRENDING DOWN: strong downtrend, sell rallies, tight stops
  - RANGING: sideways chop, mean reversion works, avoid breakouts
  - HIGH VOLATILITY: big swings both ways, reduce size, widen stops
  - CRASH: rapid sell-off, don't buy, wait for capitulation
  - RECOVERY: bouncing from crash, early longs, cautious

Different regimes need different strategies:
  - In TRENDING: momentum > mean reversion
  - In RANGING: mean reversion > momentum (our OB retests work best here)
  - In CRASH: cash is king, wait for signal
  - In HIGH VOL: reduce position size

Uses: ATR, ADX, Bollinger width, price vs MA200, drawdown from ATH
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"


@dataclass
class RegimeAnalysis:
    """Current market regime classification."""
    regime: Regime
    confidence: float       # 0-1
    adx: float              # trend strength (>25 = trending)
    atr_pct: float          # volatility as % of price
    bb_width_percentile: float  # where current BB width sits vs history
    price_vs_ma200: float   # price distance from 200-period MA (%)
    drawdown_from_ath: float  # current drawdown from all-time high (%)
    recent_return_7d: float  # 7-day return (%)
    recent_return_30d: float # 30-day return (%)

    # Strategy adjustments for this regime
    position_size_multiplier: float  # 0.0-1.5 (reduce in crash/high vol)
    preferred_strategy: str          # which strategy works best
    stop_loss_multiplier: float      # wider stops in volatile regimes
    notes: str

    def summary(self) -> str:
        return (
            f"{self.regime.value.upper()} (conf={self.confidence:.0%})\n"
            f"  ADX={self.adx:.0f} ATR={self.atr_pct:.2f}% BB={self.bb_width_percentile:.0%}\n"
            f"  vs MA200={self.price_vs_ma200:+.1f}% dd_ath={self.drawdown_from_ath:.1f}%\n"
            f"  7d={self.recent_return_7d:+.1f}% 30d={self.recent_return_30d:+.1f}%\n"
            f"  Strategy: {self.preferred_strategy}\n"
            f"  Size: {self.position_size_multiplier:.1f}x  SL: {self.stop_loss_multiplier:.1f}x\n"
            f"  {self.notes}"
        )


def detect_regime(df: pd.DataFrame) -> RegimeAnalysis:
    """Classify the current market regime from OHLCV data.

    Requires at least 200 rows for MA200 calculation.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    current = float(close.iloc[-1])

    # ADX — trend strength
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    adx = float(adx_ind.adx().iloc[-1]) if len(close) > 14 else 0

    # ATR as % of price
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    atr_pct = float(atr.iloc[-1]) / current * 100 if current > 0 else 0

    # Bollinger bandwidth percentile
    bb = ta.volatility.BollingerBands(close, window=20)
    bw = bb.bollinger_wband()
    if len(bw) >= 100:
        current_bw = float(bw.iloc[-1])
        bb_percentile = float((bw.iloc[-100:] < current_bw).mean())
    else:
        bb_percentile = 0.5

    # Price vs MA200
    if len(close) >= 200:
        ma200 = float(close.rolling(200).mean().iloc[-1])
        price_vs_ma200 = (current - ma200) / ma200 * 100
    else:
        ma50 = float(close.rolling(min(50, len(close))).mean().iloc[-1])
        price_vs_ma200 = (current - ma50) / ma50 * 100

    # Drawdown from ATH
    ath = float(high.max())
    drawdown = (ath - current) / ath * 100

    # Recent returns
    ret_7d = (current / float(close.iloc[-min(7, len(close))])) * 100 - 100 if len(close) > 7 else 0
    ret_30d = (current / float(close.iloc[-min(30, len(close))])) * 100 - 100 if len(close) > 30 else 0

    # ── Classify regime ───────────────────────────────────────────────
    regime = Regime.RANGING
    confidence = 0.5
    pos_mult = 1.0
    sl_mult = 1.0
    strategy = "ob_retest"  # default
    notes = ""

    # CRASH: rapid drawdown + high vol + bearish
    if drawdown > 20 and ret_7d < -10:
        regime = Regime.CRASH
        confidence = min(1.0, drawdown / 30)
        pos_mult = 0.25  # tiny positions
        sl_mult = 2.0    # wide stops
        strategy = "wait_for_capitulation"
        notes = "Crash mode — preserve capital, wait for capitulation signal"

    # HIGH VOLATILITY: ATR very high, BB wide
    elif atr_pct > 4 or bb_percentile > 0.9:
        regime = Regime.HIGH_VOLATILITY
        confidence = min(1.0, atr_pct / 6)
        pos_mult = 0.5
        sl_mult = 1.5
        strategy = "reduced_size_mean_reversion"
        notes = "High vol — reduce size, widen stops, mean reversion still works"

    # TRENDING UP: strong ADX + price above MA200 + positive returns
    elif adx > 25 and price_vs_ma200 > 5 and ret_30d > 5:
        regime = Regime.TRENDING_UP
        confidence = min(1.0, adx / 40)
        pos_mult = 1.2
        sl_mult = 1.0
        strategy = "momentum_with_pullback"
        notes = "Uptrend — buy pullbacks to MA20, trail stops, momentum favored"

    # TRENDING DOWN: strong ADX + price below MA200 + negative returns
    elif adx > 25 and price_vs_ma200 < -5 and ret_30d < -5:
        regime = Regime.TRENDING_DOWN
        confidence = min(1.0, adx / 40)
        pos_mult = 0.8
        sl_mult = 1.2
        strategy = "short_rallies"
        notes = "Downtrend — short rallies to MA20, tight stops, avoid catching knives"

    # RECOVERY: coming off a crash, big bounce
    elif drawdown > 15 and ret_7d > 5:
        regime = Regime.RECOVERY
        confidence = 0.6
        pos_mult = 0.7
        sl_mult = 1.3
        strategy = "cautious_longs"
        notes = "Recovery — early longs with tight stops, may be a dead cat bounce"

    # RANGING: low ADX, price near MA, small returns
    else:
        regime = Regime.RANGING
        confidence = min(1.0, (40 - adx) / 20) if adx < 40 else 0.3
        pos_mult = 1.0
        sl_mult = 1.0
        strategy = "ob_retest"
        notes = "Ranging — OB retests and FVG fills work best here"

    return RegimeAnalysis(
        regime=regime,
        confidence=round(confidence, 2),
        adx=round(adx, 1),
        atr_pct=round(atr_pct, 3),
        bb_width_percentile=round(bb_percentile, 2),
        price_vs_ma200=round(price_vs_ma200, 2),
        drawdown_from_ath=round(drawdown, 2),
        recent_return_7d=round(ret_7d, 2),
        recent_return_30d=round(ret_30d, 2),
        position_size_multiplier=round(pos_mult, 2),
        preferred_strategy=strategy,
        stop_loss_multiplier=round(sl_mult, 2),
        notes=notes,
    )
