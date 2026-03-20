"""Earnings play scanner — find stocks with upcoming earnings + historical edge.

Stocks move 5-15% on earnings. This module:
1. Scrapes upcoming earnings calendar
2. For each stock, pulls historical earnings moves (how much it moved last 4 quarters)
3. Checks current IV (implied volatility proxy via recent price range)
4. Combines with technical setup (is it at support/resistance going into earnings?)
5. Ranks opportunities by expected move vs risk

Best setups:
  - Stock at support + historically beats earnings + high IV = buy calls
  - Stock at resistance + historically misses + low IV = buy puts
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class EarningsPlay:
    """A ranked earnings play opportunity."""
    ticker: str
    company_name: str
    earnings_date: str
    days_until: int

    # Historical earnings moves
    avg_move_pct: float        # average absolute % move on earnings day
    beat_rate: float           # how often they beat estimates (0-1)
    last_4_moves: list[float]  # last 4 earnings day moves (signed)

    # Current technical setup
    current_price: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float
    rsi: float
    trend: str                 # "up", "down", "sideways"

    # Volatility
    atr_pct: float             # ATR as % of price (realized vol proxy)
    range_20d_pct: float       # 20-day high-low range as % of price

    # Scoring
    play_direction: str        # "long" | "short" | "neutral"
    score: float               # 0-1 composite opportunity score
    reasoning: str


async def scan_earnings(days_ahead: int = 14) -> list[EarningsPlay]:
    """Scan for upcoming earnings plays."""
    import ta

    # Get tickers with upcoming earnings
    # We'll use a curated list of high-move stocks + check their earnings dates
    watchlist = [
        "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "GOOGL", "META",
        "NFLX", "CRM", "SNOW", "PLTR", "COIN", "MSTR", "MARA",
        "RIOT", "SOFI", "SMCI", "ARM", "UBER", "SHOP", "RBLX",
        "CRWD", "ZS", "PANW", "DDOG", "MDB", "NET", "ABNB",
        "NKE", "FDX", "MU", "LULU", "COST", "TGT", "DG",
    ]

    plays: list[EarningsPlay] = []
    loop = asyncio.get_event_loop()

    for ticker in watchlist:
        try:
            play = await loop.run_in_executor(None, lambda t=ticker: _analyze_ticker(t, days_ahead))
            if play:
                plays.append(play)
        except Exception:
            pass

    # Sort by score
    plays.sort(key=lambda p: p.score, reverse=True)
    return plays


def _analyze_ticker(ticker: str, days_ahead: int) -> EarningsPlay | None:
    """Analyze a single ticker for earnings play potential."""
    import ta as ta_lib

    try:
        stock = yf.Ticker(ticker)

        # Check earnings date
        cal = stock.calendar
        if cal is None or cal.empty:
            return None

        # Get earnings date
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                earnings_date = str(cal["Earnings Date"].iloc[0])
            elif len(cal) > 0:
                earnings_date = str(cal.iloc[0, 0])
            else:
                return None
        else:
            return None

        # Parse date and check if within range
        try:
            from dateutil import parser as dateparser
            ed = dateparser.parse(earnings_date)
            now = datetime.now()
            days_until = (ed - now).days
            if days_until < 0 or days_until > days_ahead:
                return None
        except Exception:
            return None

        # Get historical data
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        close = df["close"]
        high = df["high"]
        low = df["low"]
        current_price = float(close.iloc[-1])

        # Historical earnings moves (from quarterly earnings)
        earnings_hist = stock.earnings_dates
        last_moves = []
        if earnings_hist is not None and not earnings_hist.empty:
            for idx in earnings_hist.index[:4]:
                date = idx.date() if hasattr(idx, "date") else idx
                # Find the closest trading day
                mask = df.index.date == date if hasattr(df.index, "date") else None
                if mask is not None and mask.any():
                    bar_idx = df.index.get_loc(df.index[mask][0])
                    if bar_idx > 0:
                        prev_close = float(close.iloc[bar_idx - 1])
                        earnings_close = float(close.iloc[bar_idx])
                        move = (earnings_close - prev_close) / prev_close * 100
                        last_moves.append(round(move, 2))

        avg_move = sum(abs(m) for m in last_moves) / len(last_moves) if last_moves else 0
        beat_rate = sum(1 for m in last_moves if m > 0) / len(last_moves) if last_moves else 0.5

        # Technical analysis
        rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
        rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50

        atr = ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_pct = float(atr.iloc[-1]) / current_price * 100 if not atr.empty else 0

        # Support/resistance
        recent_low = float(low.iloc[-20:].min())
        recent_high = float(high.iloc[-20:].max())
        dist_support = (current_price - recent_low) / current_price * 100
        dist_resistance = (recent_high - current_price) / current_price * 100
        range_20d = (recent_high - recent_low) / current_price * 100

        # Trend
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
        if current_price > ma20 > ma50:
            trend = "up"
        elif current_price < ma20 < ma50:
            trend = "down"
        else:
            trend = "sideways"

        # Score the play
        score = 0.0
        reasoning_parts = []

        # High average move = more opportunity
        if avg_move >= 8:
            score += 0.3
            reasoning_parts.append(f"avg {avg_move:.1f}% move on earnings")
        elif avg_move >= 5:
            score += 0.2
            reasoning_parts.append(f"avg {avg_move:.1f}% move")

        # Direction bias
        if beat_rate >= 0.75:
            play_direction = "long"
            score += 0.2
            reasoning_parts.append(f"beats {beat_rate:.0%} of the time")
        elif beat_rate <= 0.25:
            play_direction = "short"
            score += 0.2
            reasoning_parts.append(f"misses {1-beat_rate:.0%} of the time")
        else:
            play_direction = "neutral"

        # Technical confluence
        if play_direction == "long" and dist_support < 3 and rsi_val < 40:
            score += 0.2
            reasoning_parts.append("at support + oversold")
        elif play_direction == "short" and dist_resistance < 3 and rsi_val > 60:
            score += 0.2
            reasoning_parts.append("at resistance + overbought")

        # Sooner = more actionable
        if days_until <= 3:
            score += 0.15
            reasoning_parts.append("earnings imminent")
        elif days_until <= 7:
            score += 0.1

        # High volatility = bigger move potential
        if atr_pct >= 3:
            score += 0.1
            reasoning_parts.append("high volatility")

        name = getattr(stock.info, "get", lambda k, d="": "")("shortName", ticker)
        if callable(name):
            name = ticker

        return EarningsPlay(
            ticker=ticker,
            company_name=str(name) if name else ticker,
            earnings_date=earnings_date,
            days_until=days_until,
            avg_move_pct=round(avg_move, 2),
            beat_rate=round(beat_rate, 2),
            last_4_moves=last_moves[:4],
            current_price=current_price,
            distance_to_support_pct=round(dist_support, 2),
            distance_to_resistance_pct=round(dist_resistance, 2),
            rsi=round(rsi_val, 1),
            trend=trend,
            atr_pct=round(atr_pct, 2),
            range_20d_pct=round(range_20d, 2),
            play_direction=play_direction,
            score=round(min(1.0, score), 3),
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "moderate setup",
        )

    except Exception:
        return None
