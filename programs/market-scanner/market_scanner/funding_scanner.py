"""Funding rate scanner — find extreme funding rates for carry trades.

When funding is very negative: shorts pay longs → go long and collect funding
When funding is very positive: longs pay shorts → go short and collect funding

The funding rate is paid every 8 hours. At -1% per 8h, that's 3% per day
just from holding the position (plus or minus price movement).

This scanner:
1. Fetches all Binance futures funding rates
2. Ranks by absolute funding (most extreme = most opportunity)
3. Checks if price trend supports the funding direction
4. Calculates expected daily income from funding alone
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass
class FundingOpportunity:
    """A funding rate carry trade opportunity."""
    symbol: str
    funding_rate: float          # per 8h period
    predicted_rate: float        # predicted next funding
    annualized_pct: float        # funding rate annualized
    daily_income_pct: float      # expected daily from funding alone
    direction: str               # "long" (negative funding) | "short" (positive funding)
    price: float
    volume_24h_usd: float
    open_interest_usd: float
    risk_notes: list[str]


async def scan_funding(
    min_rate_pct: float = 0.03,
    min_volume_usd: float = 5_000_000,
) -> list[FundingOpportunity]:
    """Scan all Binance futures for extreme funding rates.

    Args:
        min_rate_pct: minimum absolute funding rate (0.03 = 0.03% per 8h)
        min_volume_usd: minimum 24h volume to ensure liquidity
    """
    opportunities: list[FundingOpportunity] = []

    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        # Get all funding rates
        resp = await client.get("https://fapi.binance.com/fapi/v1/premiumIndex")
        all_rates = resp.json()

        # Get all 24h tickers for volume
        ticker_resp = await client.get("https://fapi.binance.com/fapi/v1/ticker/24hr")
        tickers = {t["symbol"]: t for t in ticker_resp.json()}

        for item in all_rates:
            sym = item["symbol"]
            rate = float(item.get("lastFundingRate", 0))
            predicted = float(item.get("nextFundingRate", 0) or 0)
            price = float(item.get("markPrice", 0))

            if abs(rate) < min_rate_pct / 100:
                continue

            ticker = tickers.get(sym, {})
            volume = float(ticker.get("quoteVolume", 0))
            if volume < min_volume_usd:
                continue

            # Compute metrics
            daily_pct = abs(rate) * 3 * 100  # 3 funding periods per day
            annualized = daily_pct * 365
            direction = "long" if rate < 0 else "short"

            # Risk assessment
            risks = []
            if volume < 10_000_000:
                risks.append("low_volume")
            if abs(rate) > 0.01:
                risks.append("extreme_rate_may_normalize")
            price_change = float(ticker.get("priceChangePercent", 0))
            if (direction == "long" and price_change < -5) or (direction == "short" and price_change > 5):
                risks.append("price_moving_against")

            # Get OI
            try:
                oi_resp = await client.get(
                    "https://fapi.binance.com/fapi/v1/openInterest",
                    params={"symbol": sym},
                )
                oi = float(oi_resp.json().get("openInterest", 0)) * price
            except Exception:
                oi = 0

            opportunities.append(FundingOpportunity(
                symbol=sym,
                funding_rate=round(rate * 100, 4),
                predicted_rate=round(predicted * 100, 4),
                annualized_pct=round(annualized, 1),
                daily_income_pct=round(daily_pct, 3),
                direction=direction,
                price=price,
                volume_24h_usd=volume,
                open_interest_usd=oi,
                risk_notes=risks,
            ))
            await asyncio.sleep(0.02)

    # Sort by daily income potential
    opportunities.sort(key=lambda o: o.daily_income_pct, reverse=True)
    return opportunities


async def main():
    """Run the funding scanner and display results."""
    print("=" * 75)
    print("  FUNDING RATE SCANNER — Carry Trade Opportunities")
    print("=" * 75)

    opps = await scan_funding(min_rate_pct=0.02, min_volume_usd=1_000_000)

    print(f"\n  Found {len(opps)} opportunities with |rate| >= 0.02%\n")
    print(f"  {'Symbol':<14} {'Dir':<6} {'Rate/8h':>8} {'Daily':>7} {'Annual':>8} {'Vol($M)':>9} {'OI($M)':>8} {'Risks'}")
    print(f"  {'-'*75}")

    for o in opps[:20]:
        risks = ", ".join(o.risk_notes) if o.risk_notes else "clean"
        print(
            f"  {o.symbol:<14} {o.direction:<6} {o.funding_rate:>+7.3f}% "
            f"{o.daily_income_pct:>6.2f}% {o.annualized_pct:>7.0f}% "
            f"{o.volume_24h_usd/1e6:>8.1f} {o.open_interest_usd/1e6:>7.1f} {risks}"
        )

    # Summary
    if opps:
        best = opps[0]
        print(f"\n  Best opportunity: {best.direction.upper()} {best.symbol}")
        print(f"  Funding: {best.funding_rate:+.3f}% per 8h = {best.daily_income_pct:.2f}% daily = {best.annualized_pct:.0f}% annualized")
        print(f"  On a $1000 position: ${best.daily_income_pct * 10:.2f}/day from funding alone")


if __name__ == "__main__":
    asyncio.run(main())
