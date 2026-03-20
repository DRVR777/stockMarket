"""Liquidation cascade predictor.

When most traders are positioned one way (e.g. 70% long) and price starts
moving against them, their stop losses trigger a chain reaction:
  longs get liquidated → forced sells → price drops more → more liquidations

This module:
1. Monitors long/short ratios + open interest from Binance
2. Detects "crowded trade" setups where cascade risk is highest
3. When price moves against the crowd, predicts cascade magnitude
4. Emits signals before the cascade fully plays out

The best setups: high OI + extreme L/S ratio + price moving against majority
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "UNIUSDT", "AAVEUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "NEARUSDT", "SUIUSDT", "INJUSDT", "FETUSDT", "LDOUSDT",
]

# Thresholds
CROWDED_RATIO = 1.5       # L/S ratio above this = crowded long
INVERSE_RATIO = 0.67      # L/S ratio below this = crowded short
HIGH_OI_PERCENTILE = 0.7  # OI above 70th percentile = high
PRICE_TRIGGER_PCT = 0.5   # price moves 0.5% against crowd = trigger
CASCADE_SIGNAL_CHANNEL = "oracle:cascade_signal"


@dataclass
class MarketPosition:
    """Current positioning data for a single asset."""
    symbol: str
    long_short_ratio: float
    long_pct: float
    short_pct: float
    open_interest_usd: float
    volume_24h_usd: float
    oi_volume_ratio: float
    funding_rate: float
    current_price: float
    price_change_1h: float


@dataclass
class CascadeRisk:
    """Cascade risk assessment for a single asset."""
    symbol: str
    risk_score: float           # 0-1, how likely a cascade is
    crowded_direction: str      # "long" | "short" | "neutral"
    crowd_pct: float            # what % of traders are on the crowded side
    open_interest_usd: float
    trigger_price_pct: float    # how much price needs to move to start cascade
    expected_cascade_pct: float # expected price move if cascade triggers
    signals: list[str] = field(default_factory=list)


@dataclass
class CascadeSignal:
    """Actionable signal: cascade detected or imminent."""
    symbol: str
    direction: str           # "short" (if longs getting liquidated) or "long" (shorts)
    trigger: str             # what triggered it
    risk_score: float
    expected_move_pct: float
    current_price: float
    timestamp: str


async def fetch_positioning(symbols: list[str] | None = None) -> list[MarketPosition]:
    """Fetch current positioning data from Binance futures — free, no auth."""
    symbols = symbols or SYMBOLS
    positions: list[MarketPosition] = []

    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        for sym in symbols:
            try:
                # Long/short ratio
                ls_resp = await client.get(
                    "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
                    params={"symbol": sym, "period": "1h", "limit": 1},
                )
                ls_data = ls_resp.json()
                if not ls_data or not isinstance(ls_data, list):
                    continue
                ls = ls_data[0]

                # Open interest
                oi_resp = await client.get(
                    "https://fapi.binance.com/fapi/v1/openInterest",
                    params={"symbol": sym},
                )
                oi_data = oi_resp.json()

                # 24h ticker
                ticker_resp = await client.get(
                    "https://fapi.binance.com/fapi/v1/ticker/24hr",
                    params={"symbol": sym},
                )
                ticker = ticker_resp.json()

                # Funding rate
                funding_resp = await client.get(
                    "https://fapi.binance.com/fapi/v1/premiumIndex",
                    params={"symbol": sym},
                )
                funding = funding_resp.json()

                price = float(ticker.get("lastPrice", 0))
                oi = float(oi_data.get("openInterest", 0))
                volume = float(ticker.get("quoteVolume", 0))

                positions.append(MarketPosition(
                    symbol=sym,
                    long_short_ratio=float(ls.get("longShortRatio", 1)),
                    long_pct=float(ls.get("longAccount", 0.5)) * 100,
                    short_pct=float(ls.get("shortAccount", 0.5)) * 100,
                    open_interest_usd=oi * price,
                    volume_24h_usd=volume,
                    oi_volume_ratio=(oi * price) / volume if volume > 0 else 0,
                    funding_rate=float(funding.get("lastFundingRate", 0)),
                    current_price=price,
                    price_change_1h=float(ticker.get("priceChangePercent", 0)),
                ))
                await asyncio.sleep(0.1)
            except Exception:
                pass

    return positions


def assess_cascade_risk(positions: list[MarketPosition]) -> list[CascadeRisk]:
    """Assess cascade risk for each asset based on positioning data."""
    if not positions:
        return []

    # Compute OI percentiles for relative comparison
    oi_values = [p.open_interest_usd for p in positions]
    oi_75th = np.percentile(oi_values, 75) if oi_values else 0

    risks: list[CascadeRisk] = []

    for pos in positions:
        signals: list[str] = []
        risk = 0.0

        # 1. Crowding score
        if pos.long_short_ratio >= CROWDED_RATIO:
            crowded = "long"
            crowd_pct = pos.long_pct
            risk += 0.25
            signals.append(f"crowded_long_{pos.long_pct:.0f}pct")
        elif pos.long_short_ratio <= INVERSE_RATIO:
            crowded = "short"
            crowd_pct = pos.short_pct
            risk += 0.25
            signals.append(f"crowded_short_{pos.short_pct:.0f}pct")
        else:
            crowded = "neutral"
            crowd_pct = max(pos.long_pct, pos.short_pct)

        # 2. High OI score
        if pos.open_interest_usd >= oi_75th:
            risk += 0.15
            signals.append("high_oi")

        # 3. Funding rate divergence (positive funding + longs crowded = danger)
        if crowded == "long" and pos.funding_rate > 0.0005:
            risk += 0.15
            signals.append("funding_against_crowd")
        elif crowded == "short" and pos.funding_rate < -0.0005:
            risk += 0.15
            signals.append("funding_against_crowd")

        # 4. OI/Volume ratio (high = lots of open positions relative to trading = illiquid)
        if pos.oi_volume_ratio > 1.0:
            risk += 0.1
            signals.append("illiquid_oi")

        # 5. Price already moving against the crowd
        if crowded == "long" and pos.price_change_1h < -PRICE_TRIGGER_PCT:
            risk += 0.25
            signals.append("price_moving_against_longs")
        elif crowded == "short" and pos.price_change_1h > PRICE_TRIGGER_PCT:
            risk += 0.25
            signals.append("price_moving_against_shorts")

        # Expected cascade magnitude
        # More crowded + more OI = bigger cascade
        crowd_factor = (crowd_pct - 50) / 50 if crowd_pct > 50 else 0  # 0-1
        oi_factor = min(1.0, pos.open_interest_usd / (oi_75th or 1))
        expected_cascade = crowd_factor * oi_factor * 3.0  # max ~3% cascade

        # Trigger price: how much more does price need to move
        if crowded == "long":
            trigger = max(0, PRICE_TRIGGER_PCT - abs(pos.price_change_1h))
        elif crowded == "short":
            trigger = max(0, PRICE_TRIGGER_PCT - abs(pos.price_change_1h))
        else:
            trigger = PRICE_TRIGGER_PCT

        risks.append(CascadeRisk(
            symbol=pos.symbol,
            risk_score=min(1.0, risk),
            crowded_direction=crowded,
            crowd_pct=crowd_pct,
            open_interest_usd=pos.open_interest_usd,
            trigger_price_pct=round(trigger, 2),
            expected_cascade_pct=round(expected_cascade, 2),
            signals=signals,
        ))

    # Sort by risk score
    risks.sort(key=lambda r: r.risk_score, reverse=True)
    return risks


class CascadePredictor:
    """Monitor positioning and emit cascade warning signals."""

    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis = redis_client

    async def scan(self) -> list[CascadeRisk]:
        """Run a single scan cycle."""
        positions = await fetch_positioning()
        risks = assess_cascade_risk(positions)

        # Emit signals for high-risk assets
        for risk in risks:
            if risk.risk_score >= 0.5 and self._redis:
                direction = "short" if risk.crowded_direction == "long" else "long"
                signal = CascadeSignal(
                    symbol=risk.symbol,
                    direction=direction,
                    trigger=", ".join(risk.signals),
                    risk_score=risk.risk_score,
                    expected_move_pct=risk.expected_cascade_pct,
                    current_price=0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                await self._redis.publish(
                    CASCADE_SIGNAL_CHANNEL,
                    json.dumps({
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "risk_score": signal.risk_score,
                        "expected_move_pct": signal.expected_move_pct,
                        "trigger": signal.trigger,
                        "timestamp": signal.timestamp,
                    }),
                )

        return risks

    async def monitor(self, interval_seconds: int = 300) -> None:
        """Continuously monitor for cascade risk."""
        while True:
            risks = await self.scan()
            high_risk = [r for r in risks if r.risk_score >= 0.4]
            if high_risk:
                logger.info(
                    "CascadePredictor: %d high-risk assets detected",
                    len(high_risk),
                )
            await asyncio.sleep(interval_seconds)
