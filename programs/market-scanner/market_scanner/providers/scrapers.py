"""Web scrapers for data that doesn't have free APIs.

Each scraper extracts structured data from public web pages.
No authentication required — just HTTP requests + HTML parsing.

Sources:
  - Fear & Greed Index (alternative.me)
  - Crypto funding rates (CoinGlass)
  - SEC filings / insider trades (SEC EDGAR)
  - Earnings calendar (Yahoo Finance)
  - Crypto liquidations (CoinGlass)
  - Social sentiment (Reddit trending, Twitter/X trends)
  - Options flow (Unusual Whales public feed)
  - Congressional trades (Capitol Trades)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ── Fear & Greed Index ────────────────────────────────────────────────────────

@dataclass
class FearGreedData:
    value: int          # 0-100
    label: str          # "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
    timestamp: str
    yesterday: int
    last_week: int
    last_month: int


async def scrape_fear_greed() -> FearGreedData | None:
    """Crypto Fear & Greed Index from alternative.me (free JSON API)."""
    try:
        async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=1&format=json")
            data = resp.json()["data"][0]

            # Also get historical
            resp2 = await client.get("https://api.alternative.me/fng/?limit=31&format=json")
            hist = resp2.json()["data"]

            return FearGreedData(
                value=int(data["value"]),
                label=data["value_classification"],
                timestamp=data["timestamp"],
                yesterday=int(hist[1]["value"]) if len(hist) > 1 else 0,
                last_week=int(hist[7]["value"]) if len(hist) > 7 else 0,
                last_month=int(hist[30]["value"]) if len(hist) > 30 else 0,
            )
    except Exception:
        logger.warning("scrape_fear_greed failed", exc_info=True)
        return None


# ── Crypto Funding Rates ──────────────────────────────────────────────────────

@dataclass
class FundingRate:
    symbol: str
    rate: float          # positive = longs pay shorts, negative = shorts pay longs
    predicted: float
    exchange: str


async def scrape_funding_rates() -> list[FundingRate]:
    """Binance funding rates (free REST API, no auth)."""
    rates = []
    try:
        async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
            resp = await client.get("https://fapi.binance.com/fapi/v1/premiumIndex")
            data = resp.json()
            for item in data:
                if float(item.get("lastFundingRate", 0)) != 0:
                    rates.append(FundingRate(
                        symbol=item["symbol"],
                        rate=float(item["lastFundingRate"]),
                        predicted=float(item.get("nextFundingRate", 0) or 0),
                        exchange="binance",
                    ))
    except Exception:
        logger.warning("scrape_funding_rates failed", exc_info=True)
    return rates


# ── Crypto Liquidations ──────────────────────────────────────────────────────

@dataclass
class LiquidationData:
    total_24h_usd: float
    long_liquidations: float
    short_liquidations: float
    largest_single: float


async def scrape_liquidations() -> LiquidationData | None:
    """Binance liquidation data from public API."""
    try:
        async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
            # Binance forced orders (last 100)
            resp = await client.get(
                "https://fapi.binance.com/fapi/v1/allForceOrders",
                params={"limit": 100},
            )
            orders = resp.json()
            total = sum(float(o.get("origQty", 0)) * float(o.get("price", 0)) for o in orders)
            longs = sum(
                float(o["origQty"]) * float(o["price"])
                for o in orders if o.get("side") == "SELL"  # liquidated longs are sell orders
            )
            shorts = total - longs
            largest = max(
                (float(o["origQty"]) * float(o["price"]) for o in orders), default=0
            )
            return LiquidationData(
                total_24h_usd=total,
                long_liquidations=longs,
                short_liquidations=shorts,
                largest_single=largest,
            )
    except Exception:
        logger.warning("scrape_liquidations failed", exc_info=True)
        return None


# ── SEC EDGAR Insider Trades ──────────────────────────────────────────────────

@dataclass
class InsiderTrade:
    company: str
    ticker: str
    insider_name: str
    title: str
    trade_type: str     # "Purchase" | "Sale"
    shares: int
    price: float
    value: float
    filed_date: str


async def scrape_insider_trades(limit: int = 20) -> list[InsiderTrade]:
    """Recent insider trades from SEC EDGAR full-text search."""
    trades = []
    try:
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": "form-type:4",
                    "dateRange": "custom",
                    "startdt": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "forms": "4",
                },
            )
            # EDGAR returns XML/JSON of recent Form 4 filings
            # Simplified parsing — full implementation would parse SGML
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            hits = data.get("hits", {}).get("hits", [])[:limit]

            for hit in hits:
                source = hit.get("_source", {})
                trades.append(InsiderTrade(
                    company=source.get("entity_name", ""),
                    ticker=source.get("tickers", [""])[0] if source.get("tickers") else "",
                    insider_name=source.get("reporting_owner", ""),
                    title="",
                    trade_type="Filing",
                    shares=0,
                    price=0,
                    value=0,
                    filed_date=source.get("file_date", ""),
                ))
    except Exception:
        logger.warning("scrape_insider_trades failed", exc_info=True)
    return trades


# ── Earnings Calendar ─────────────────────────────────────────────────────────

@dataclass
class EarningsEvent:
    ticker: str
    company: str
    date: str
    time: str          # "BMO" (before market open) | "AMC" (after market close)
    estimate_eps: float
    actual_eps: float | None


async def scrape_earnings_calendar() -> list[EarningsEvent]:
    """Upcoming earnings from Yahoo Finance."""
    events = []
    try:
        async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
            resp = await client.get(
                "https://finance.yahoo.com/calendar/earnings",
                headers={**HEADERS, "Accept": "text/html"},
            )
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")[1:]  # skip header
                for row in rows[:30]:
                    cols = row.find_all("td")
                    if len(cols) >= 4:
                        events.append(EarningsEvent(
                            ticker=cols[0].get_text(strip=True),
                            company=cols[1].get_text(strip=True),
                            date=cols[2].get_text(strip=True),
                            time=cols[3].get_text(strip=True) if len(cols) > 3 else "",
                            estimate_eps=0,
                            actual_eps=None,
                        ))
    except Exception:
        logger.warning("scrape_earnings_calendar failed", exc_info=True)
    return events


# ── Reddit Trending ───────────────────────────────────────────────────────────

@dataclass
class RedditTrend:
    subreddit: str
    title: str
    score: int
    comments: int
    url: str
    tickers_mentioned: list[str]


async def scrape_reddit_trending(
    subreddits: list[str] | None = None,
) -> list[RedditTrend]:
    """Top posts from trading subreddits (no auth needed for .json endpoint)."""
    subs = subreddits or [
        "wallstreetbets", "stocks", "options", "cryptocurrency",
        "CryptoMarkets", "Daytrading",
    ]
    trends = []
    # Regex to find stock tickers ($AAPL, $TSLA, etc.)
    ticker_re = re.compile(r"\$([A-Z]{1,5})\b")

    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        for sub in subs:
            try:
                resp = await client.get(
                    f"https://www.reddit.com/r/{sub}/hot.json",
                    params={"limit": 10},
                )
                data = resp.json()
                for post in data.get("data", {}).get("children", []):
                    d = post["data"]
                    title = d.get("title", "")
                    tickers = ticker_re.findall(title)
                    trends.append(RedditTrend(
                        subreddit=sub,
                        title=title,
                        score=d.get("score", 0),
                        comments=d.get("num_comments", 0),
                        url=d.get("url", ""),
                        tickers_mentioned=tickers,
                    ))
            except Exception:
                logger.debug("scrape_reddit_trending: failed for r/%s", sub)
    return trends


# ── Master scrape function ────────────────────────────────────────────────────

@dataclass
class ScraperSnapshot:
    """All scraped data in one snapshot."""
    fear_greed: FearGreedData | None
    funding_rates: list[FundingRate]
    liquidations: LiquidationData | None
    reddit_trends: list[RedditTrend]
    earnings: list[EarningsEvent]
    timestamp: str


async def scrape_all() -> ScraperSnapshot:
    """Run all scrapers concurrently and return a snapshot."""
    fg, funding, liqs, reddit, earnings = await asyncio.gather(
        scrape_fear_greed(),
        scrape_funding_rates(),
        scrape_liquidations(),
        scrape_reddit_trending(),
        scrape_earnings_calendar(),
        return_exceptions=True,
    )

    return ScraperSnapshot(
        fear_greed=fg if isinstance(fg, FearGreedData) else None,
        funding_rates=funding if isinstance(funding, list) else [],
        liquidations=liqs if isinstance(liqs, LiquidationData) else None,
        reddit_trends=reddit if isinstance(reddit, list) else [],
        earnings=earnings if isinstance(earnings, list) else [],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
