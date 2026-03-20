"""Smart Money Concepts (SMC) analysis engine.

Implements institutional order flow analysis:
  - Market Structure: BOS, CHoCH, MSS
  - Order Blocks (OB): bullish and bearish institutional zones
  - Fair Value Gaps (FVG): imbalance zones
  - Liquidity: equal highs/lows, liquidity sweeps
  - Premium/Discount zones via Fibonacci
  - Displacement detection (strong momentum candles)

Works with any OHLCV DataFrame (crypto, stocks, any timeframe).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SwingPoint:
    """A swing high or swing low in price structure."""
    index: int
    price: float
    type: str  # "high" | "low"
    timestamp: object = None


@dataclass
class FairValueGap:
    """An imbalance zone where price moved too fast."""
    start_index: int
    high: float       # top of the gap
    low: float        # bottom of the gap
    direction: str    # "bullish" (gap up) | "bearish" (gap down)
    filled: bool = False
    timestamp: object = None


@dataclass
class OrderBlock:
    """Institutional accumulation/distribution zone."""
    index: int
    high: float
    low: float
    direction: str    # "bullish" (demand zone) | "bearish" (supply zone)
    tested: bool = False
    timestamp: object = None


@dataclass
class LiquidityLevel:
    """Equal highs/lows that act as liquidity targets."""
    price: float
    type: str         # "equal_highs" | "equal_lows" | "swing_high" | "swing_low"
    count: int = 1    # how many times price touched this level
    swept: bool = False


@dataclass
class StructureBreak:
    """Break of Structure (BOS) or Change of Character (CHoCH)."""
    index: int
    price: float
    type: str         # "bos_bullish" | "bos_bearish" | "choch_bullish" | "choch_bearish"
    timestamp: object = None


@dataclass
class SMCAnalysis:
    """Complete SMC analysis result for an asset."""
    symbol: str
    asset_type: str
    bias: Bias
    confidence: float            # 0.0-1.0

    # Structure
    trend: str                   # "uptrend" | "downtrend" | "ranging"
    structure_breaks: list[StructureBreak] = field(default_factory=list)
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)

    # Order flow
    order_blocks: list[OrderBlock] = field(default_factory=list)
    fair_value_gaps: list[FairValueGap] = field(default_factory=list)
    liquidity_levels: list[LiquidityLevel] = field(default_factory=list)

    # Zones
    premium_zone: tuple[float, float] = (0, 0)   # (low, high) — above equilibrium
    discount_zone: tuple[float, float] = (0, 0)   # (low, high) — below equilibrium
    equilibrium: float = 0.0

    # Trade setup
    entry_zone: Optional[tuple[float, float]] = None  # (low, high) of ideal entry
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    setup_type: str = ""         # "ob_retest", "fvg_fill", "liquidity_sweep", "bos_pullback"

    # Signals detected
    signals: list[str] = field(default_factory=list)
    current_price: float = 0.0


# ── Swing point detection ─────────────────────────────────────────────────────

def find_swing_points(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 5,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """Detect swing highs and swing lows using a rolling window."""
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for i in range(lookback, len(high) - lookback):
        # Swing high: highest in window
        window_high = high.iloc[i - lookback : i + lookback + 1]
        if float(high.iloc[i]) == float(window_high.max()):
            swing_highs.append(SwingPoint(
                index=i,
                price=float(high.iloc[i]),
                type="high",
                timestamp=high.index[i] if hasattr(high.index, '__getitem__') else None,
            ))

        # Swing low: lowest in window
        window_low = low.iloc[i - lookback : i + lookback + 1]
        if float(low.iloc[i]) == float(window_low.min()):
            swing_lows.append(SwingPoint(
                index=i,
                price=float(low.iloc[i]),
                type="low",
                timestamp=low.index[i] if hasattr(low.index, '__getitem__') else None,
            ))

    return swing_highs, swing_lows


# ── Market structure ──────────────────────────────────────────────────────────

def detect_structure(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    close: pd.Series,
) -> tuple[str, list[StructureBreak]]:
    """Detect BOS (Break of Structure) and CHoCH (Change of Character).

    BOS: continuation — higher high in uptrend or lower low in downtrend.
    CHoCH: reversal — first lower low in uptrend or first higher high in downtrend.
    """
    breaks: list[StructureBreak] = []

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "ranging", breaks

    # Determine current trend from last 3 swing points
    recent_highs = swing_highs[-3:]
    recent_lows = swing_lows[-3:]

    hh = all(recent_highs[i].price < recent_highs[i + 1].price for i in range(len(recent_highs) - 1))
    hl = all(recent_lows[i].price < recent_lows[i + 1].price for i in range(len(recent_lows) - 1))
    lh = all(recent_highs[i].price > recent_highs[i + 1].price for i in range(len(recent_highs) - 1))
    ll = all(recent_lows[i].price > recent_lows[i + 1].price for i in range(len(recent_lows) - 1))

    if hh and hl:
        trend = "uptrend"
    elif lh and ll:
        trend = "downtrend"
    else:
        trend = "ranging"

    # Detect structure breaks
    current_price = float(close.iloc[-1])

    if len(swing_highs) >= 2:
        last_sh = swing_highs[-1]
        prev_sh = swing_highs[-2]

        # BOS bullish: price breaks above last swing high in uptrend
        if current_price > last_sh.price and trend == "uptrend":
            breaks.append(StructureBreak(
                index=len(close) - 1,
                price=last_sh.price,
                type="bos_bullish",
            ))

        # CHoCH bullish: in downtrend, price breaks above a lower high
        if current_price > last_sh.price and trend == "downtrend":
            breaks.append(StructureBreak(
                index=len(close) - 1,
                price=last_sh.price,
                type="choch_bullish",
            ))

    if len(swing_lows) >= 2:
        last_sl = swing_lows[-1]

        # BOS bearish: price breaks below last swing low in downtrend
        if current_price < last_sl.price and trend == "downtrend":
            breaks.append(StructureBreak(
                index=len(close) - 1,
                price=last_sl.price,
                type="bos_bearish",
            ))

        # CHoCH bearish: in uptrend, price breaks below a higher low
        if current_price < last_sl.price and trend == "uptrend":
            breaks.append(StructureBreak(
                index=len(close) - 1,
                price=last_sl.price,
                type="choch_bearish",
            ))

    return trend, breaks


# ── Fair Value Gaps ───────────────────────────────────────────────────────────

def find_fvg(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> list[FairValueGap]:
    """Detect Fair Value Gaps (3-candle imbalance zones).

    Bullish FVG: candle[i+2].low > candle[i].high (gap between candle 1 and 3)
    Bearish FVG: candle[i].low > candle[i+2].high
    """
    fvgs: list[FairValueGap] = []

    for i in range(len(high) - 2):
        # Bullish FVG: strong up move leaves a gap
        if float(low.iloc[i + 2]) > float(high.iloc[i]):
            gap_low = float(high.iloc[i])
            gap_high = float(low.iloc[i + 2])
            # Check if gap has been filled
            filled = any(float(low.iloc[j]) <= gap_low for j in range(i + 3, len(low)))
            fvgs.append(FairValueGap(
                start_index=i,
                high=gap_high,
                low=gap_low,
                direction="bullish",
                filled=filled,
                timestamp=high.index[i + 1] if hasattr(high.index, '__getitem__') else None,
            ))

        # Bearish FVG: strong down move leaves a gap
        if float(low.iloc[i]) > float(high.iloc[i + 2]):
            gap_high = float(low.iloc[i])
            gap_low = float(high.iloc[i + 2])
            filled = any(float(high.iloc[j]) >= gap_high for j in range(i + 3, len(high)))
            fvgs.append(FairValueGap(
                start_index=i,
                high=gap_high,
                low=gap_low,
                direction="bearish",
                filled=filled,
                timestamp=high.index[i + 1] if hasattr(high.index, '__getitem__') else None,
            ))

    return fvgs


# ── Order Blocks ──────────────────────────────────────────────────────────────

def find_order_blocks(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> list[OrderBlock]:
    """Detect order blocks — the last opposing candle before a strong move.

    Bullish OB: last bearish candle before a swing low (demand zone).
    Bearish OB: last bullish candle before a swing high (supply zone).
    """
    obs: list[OrderBlock] = []

    for sl in swing_lows:
        idx = sl.index
        # Look back for the last bearish candle before this swing low
        for j in range(idx, max(idx - 5, 0), -1):
            if float(close.iloc[j]) < float(open_.iloc[j]):  # bearish candle
                obs.append(OrderBlock(
                    index=j,
                    high=float(high.iloc[j]),
                    low=float(low.iloc[j]),
                    direction="bullish",
                    timestamp=high.index[j] if hasattr(high.index, '__getitem__') else None,
                ))
                break

    for sh in swing_highs:
        idx = sh.index
        for j in range(idx, max(idx - 5, 0), -1):
            if float(close.iloc[j]) > float(open_.iloc[j]):  # bullish candle
                obs.append(OrderBlock(
                    index=j,
                    high=float(high.iloc[j]),
                    low=float(low.iloc[j]),
                    direction="bearish",
                    timestamp=high.index[j] if hasattr(high.index, '__getitem__') else None,
                ))
                break

    return obs


# ── Liquidity levels ──────────────────────────────────────────────────────────

def find_liquidity(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    tolerance_pct: float = 0.002,
) -> list[LiquidityLevel]:
    """Detect equal highs/lows (liquidity pools) and key swing levels."""
    levels: list[LiquidityLevel] = []

    # Equal highs — multiple swing highs at similar price
    for i, sh1 in enumerate(swing_highs):
        count = 1
        for sh2 in swing_highs[i + 1 :]:
            if abs(sh1.price - sh2.price) / sh1.price < tolerance_pct:
                count += 1
        if count >= 2:
            levels.append(LiquidityLevel(
                price=sh1.price,
                type="equal_highs",
                count=count,
            ))

    # Equal lows
    for i, sl1 in enumerate(swing_lows):
        count = 1
        for sl2 in swing_lows[i + 1 :]:
            if abs(sl1.price - sl2.price) / sl1.price < tolerance_pct:
                count += 1
        if count >= 2:
            levels.append(LiquidityLevel(
                price=sl1.price,
                type="equal_lows",
                count=count,
            ))

    # Deduplicate by price proximity
    unique: list[LiquidityLevel] = []
    for lev in levels:
        if not any(abs(lev.price - u.price) / lev.price < tolerance_pct for u in unique):
            unique.append(lev)

    return unique


# ── Premium / Discount zones ─────────────────────────────────────────────────

def compute_zones(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Compute equilibrium and premium/discount zones.

    Premium zone: above 50% Fib (expensive — look to sell)
    Discount zone: below 50% Fib (cheap — look to buy)
    """
    if not swing_highs or not swing_lows:
        return 0.0, (0, 0), (0, 0)

    highest = max(sh.price for sh in swing_highs[-5:])
    lowest = min(sl.price for sl in swing_lows[-5:])
    range_ = highest - lowest
    equilibrium = lowest + range_ * 0.5

    premium = (equilibrium, highest)
    discount = (lowest, equilibrium)

    return equilibrium, premium, discount


# ── Displacement detection ────────────────────────────────────────────────────

def detect_displacement(
    open_: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    threshold_pct: float = 0.02,
) -> list[dict]:
    """Detect displacement candles — strong momentum moves.

    A displacement is a candle with body > threshold_pct of price
    and engulfs previous candle(s).
    """
    displacements = []
    for i in range(1, len(close)):
        body = abs(float(close.iloc[i]) - float(open_.iloc[i]))
        candle_range = float(high.iloc[i]) - float(low.iloc[i])
        if candle_range == 0:
            continue

        body_pct = body / float(close.iloc[i])
        body_ratio = body / candle_range

        if body_pct >= threshold_pct and body_ratio >= 0.6:
            direction = "bullish" if float(close.iloc[i]) > float(open_.iloc[i]) else "bearish"
            displacements.append({
                "index": i,
                "direction": direction,
                "body_pct": round(body_pct * 100, 2),
            })

    return displacements


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_smc(
    df: pd.DataFrame,
    symbol: str,
    asset_type: str,
) -> Optional[SMCAnalysis]:
    """Run full Smart Money Concepts analysis.

    Requires at least 30 rows of OHLCV data.
    """
    if len(df) < 30:
        return None

    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    current_price = float(close.iloc[-1])

    # 1. Find swing points
    swing_highs, swing_lows = find_swing_points(high, low, lookback=5)
    if not swing_highs or not swing_lows:
        return None

    # 2. Market structure
    trend, structure_breaks = detect_structure(swing_highs, swing_lows, close)

    # 3. Fair value gaps
    fvgs = find_fvg(high, low, close)
    unfilled_fvgs = [f for f in fvgs if not f.filled]

    # 4. Order blocks
    obs = find_order_blocks(open_, high, low, close, swing_highs, swing_lows)

    # 5. Liquidity levels
    liquidity = find_liquidity(swing_highs, swing_lows)

    # 6. Premium/Discount zones
    equilibrium, premium_zone, discount_zone = compute_zones(swing_highs, swing_lows)

    # 7. Displacement
    displacements = detect_displacement(open_, close, high, low)
    recent_displacements = [d for d in displacements if d["index"] >= len(close) - 5]

    # ── Build trade setup ──────────────────────────────────────────────────
    signals: list[str] = []
    confidence = 0.0
    bias = Bias.NEUTRAL
    setup_type = ""
    entry_zone = None
    stop_loss = None
    take_profit = None

    # Structure signals
    for brk in structure_breaks:
        signals.append(brk.type)
        if "bullish" in brk.type:
            confidence += 0.15
        elif "bearish" in brk.type:
            confidence += 0.15

    if trend == "uptrend":
        confidence += 0.1
        signals.append("uptrend")
    elif trend == "downtrend":
        confidence += 0.1
        signals.append("downtrend")

    # FVG signals — unfilled gaps near current price are magnets
    for fvg in unfilled_fvgs[-3:]:  # last 3 unfilled
        gap_mid = (fvg.high + fvg.low) / 2
        distance_pct = abs(current_price - gap_mid) / current_price
        if distance_pct < 0.05:  # within 5%
            signals.append(f"fvg_{fvg.direction}_nearby")
            confidence += 0.1
            if fvg.direction == "bullish" and current_price <= fvg.high:
                entry_zone = (fvg.low, fvg.high)
                setup_type = "fvg_fill"
            elif fvg.direction == "bearish" and current_price >= fvg.low:
                entry_zone = (fvg.low, fvg.high)
                setup_type = "fvg_fill"

    # Order block signals — price retesting an OB is a high-probability setup
    for ob in obs[-5:]:
        if ob.direction == "bullish" and ob.low <= current_price <= ob.high:
            signals.append("bullish_ob_retest")
            confidence += 0.2
            entry_zone = (ob.low, ob.high)
            setup_type = "ob_retest"
        elif ob.direction == "bearish" and ob.low <= current_price <= ob.high:
            signals.append("bearish_ob_retest")
            confidence += 0.2
            entry_zone = (ob.low, ob.high)
            setup_type = "ob_retest"

    # Liquidity sweep signals
    for liq in liquidity:
        distance_pct = abs(current_price - liq.price) / current_price
        if distance_pct < 0.01:  # within 1%
            if liq.type == "equal_highs":
                signals.append("liquidity_sweep_highs")
                confidence += 0.15
            elif liq.type == "equal_lows":
                signals.append("liquidity_sweep_lows")
                confidence += 0.15

    # Premium/Discount zone
    if discount_zone[0] <= current_price <= discount_zone[1]:
        signals.append("in_discount_zone")
        confidence += 0.1
    elif premium_zone[0] <= current_price <= premium_zone[1]:
        signals.append("in_premium_zone")
        confidence += 0.1

    # Displacement signals
    if recent_displacements:
        last_disp = recent_displacements[-1]
        signals.append(f"displacement_{last_disp['direction']}")
        confidence += 0.15

    # ── Determine bias ─────────────────────────────────────────────────────
    bull_signals = sum(1 for s in signals if any(b in s for b in ["bullish", "uptrend", "discount", "sweep_lows"]))
    bear_signals = sum(1 for s in signals if any(b in s for b in ["bearish", "downtrend", "premium", "sweep_highs"]))

    if bull_signals > bear_signals:
        bias = Bias.BULLISH
        # Set trade params for long
        if not stop_loss and swing_lows:
            stop_loss = swing_lows[-1].price * 0.99
        if not take_profit and swing_highs:
            take_profit = swing_highs[-1].price
    elif bear_signals > bull_signals:
        bias = Bias.BEARISH
        # Set trade params for short
        if not stop_loss and swing_highs:
            stop_loss = swing_highs[-1].price * 1.01
        if not take_profit and swing_lows:
            take_profit = swing_lows[-1].price
    else:
        bias = Bias.NEUTRAL

    confidence = min(1.0, confidence)

    return SMCAnalysis(
        symbol=symbol,
        asset_type=asset_type,
        bias=bias,
        confidence=round(confidence, 3),
        trend=trend,
        structure_breaks=structure_breaks[-3:],
        swing_highs=swing_highs[-5:],
        swing_lows=swing_lows[-5:],
        order_blocks=obs[-5:],
        fair_value_gaps=unfilled_fvgs[-5:],
        liquidity_levels=liquidity,
        premium_zone=premium_zone,
        discount_zone=discount_zone,
        equilibrium=round(equilibrium, 4),
        entry_zone=entry_zone,
        stop_loss=round(stop_loss, 4) if stop_loss else None,
        take_profit=round(take_profit, 4) if take_profit else None,
        setup_type=setup_type,
        signals=signals,
        current_price=current_price,
    )
