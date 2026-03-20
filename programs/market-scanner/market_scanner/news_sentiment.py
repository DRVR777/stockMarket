"""News sentiment classifier — scrape headlines, classify with Gemini.

Pulls latest headlines for any ticker from Google News RSS (free, no auth),
classifies each as bullish/bearish/neutral using Gemini, and scores overall
sentiment for the asset.

Usage::

    classifier = NewsSentiment()
    result = await classifier.analyze("TSLA")
    print(result.score, result.headlines)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
}


@dataclass
class HeadlineSentiment:
    title: str
    source: str
    published: str
    sentiment: str    # "bullish" | "bearish" | "neutral"
    confidence: float  # 0-1


@dataclass
class SentimentResult:
    symbol: str
    score: float              # -1.0 (max bearish) to +1.0 (max bullish)
    bullish_count: int
    bearish_count: int
    neutral_count: int
    headlines: list[HeadlineSentiment] = field(default_factory=list)
    analyzed_at: str = ""


async def fetch_google_news(query: str, limit: int = 15) -> list[dict]:
    """Fetch headlines from Google News RSS — free, no auth, no limit."""
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    headlines = []
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        try:
            resp = await client.get(url)
            root = ElementTree.fromstring(resp.text)
            for item in root.findall(".//item")[:limit]:
                title = item.findtext("title", "")
                source = item.findtext("source", "")
                pub_date = item.findtext("pubDate", "")
                # Clean HTML entities
                title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                headlines.append({
                    "title": title,
                    "source": source,
                    "published": pub_date,
                })
        except Exception:
            logger.warning("fetch_google_news failed for %s", query, exc_info=True)
    return headlines


async def classify_headlines_batch(
    headlines: list[dict],
    symbol: str,
) -> list[HeadlineSentiment]:
    """Classify multiple headlines using Gemini in one batch call."""
    if not headlines:
        return []

    try:
        from oracle_shared.providers import get_llm
        llm = get_llm()
    except Exception:
        # Fallback: keyword-based classification
        return _keyword_classify(headlines)

    titles = "\n".join(f'{i+1}. {h["title"]}' for i, h in enumerate(headlines))
    prompt = (
        f"For the stock/crypto ticker {symbol}, classify each headline as "
        f"bullish, bearish, or neutral. Return ONLY JSON array:\n"
        f'[{{"index": 1, "sentiment": "bullish|bearish|neutral", "confidence": 0.0-1.0}}, ...]\n\n'
        f"Headlines:\n{titles}"
    )

    try:
        result = await llm.generate_json(prompt, max_tokens=500)
        if not isinstance(result, list):
            result = result.get("classifications", result.get("headlines", []))

        classified = []
        for item in result:
            idx = item.get("index", 0) - 1
            if 0 <= idx < len(headlines):
                h = headlines[idx]
                classified.append(HeadlineSentiment(
                    title=h["title"],
                    source=h.get("source", ""),
                    published=h.get("published", ""),
                    sentiment=item.get("sentiment", "neutral"),
                    confidence=float(item.get("confidence", 0.5)),
                ))
        return classified
    except Exception:
        logger.debug("Gemini classification failed, using keywords", exc_info=True)
        return _keyword_classify(headlines)


def _keyword_classify(headlines: list[dict]) -> list[HeadlineSentiment]:
    """Fast keyword-based sentiment when Gemini is unavailable."""
    bullish_words = {
        "surge", "soar", "rally", "jump", "gain", "rise", "bull", "record",
        "breakout", "upgrade", "buy", "growth", "profit", "beat", "exceed",
        "all-time high", "boom", "rocket", "moon", "pump",
    }
    bearish_words = {
        "crash", "plunge", "drop", "fall", "sink", "bear", "decline", "sell",
        "downturn", "warning", "fear", "loss", "miss", "cut", "lawsuit",
        "investigation", "fraud", "dump", "tank", "collapse", "smuggl",
    }

    results = []
    for h in headlines:
        title_lower = h["title"].lower()
        bull = sum(1 for w in bullish_words if w in title_lower)
        bear = sum(1 for w in bearish_words if w in title_lower)

        if bull > bear:
            sentiment = "bullish"
            conf = min(0.9, 0.5 + bull * 0.15)
        elif bear > bull:
            sentiment = "bearish"
            conf = min(0.9, 0.5 + bear * 0.15)
        else:
            sentiment = "neutral"
            conf = 0.5

        results.append(HeadlineSentiment(
            title=h["title"],
            source=h.get("source", ""),
            published=h.get("published", ""),
            sentiment=sentiment,
            confidence=conf,
        ))
    return results


class NewsSentiment:
    """Analyze news sentiment for any ticker."""

    async def analyze(
        self,
        symbol: str,
        use_gemini: bool = True,
        limit: int = 15,
    ) -> SentimentResult:
        """Fetch and classify news for a symbol.

        Returns SentimentResult with score from -1.0 to +1.0.
        """
        # Fetch headlines
        # Try both ticker and common name
        queries = [f"{symbol} stock", symbol]
        all_headlines: list[dict] = []
        for q in queries:
            headlines = await fetch_google_news(q, limit=limit)
            all_headlines.extend(headlines)
            if len(all_headlines) >= limit:
                break

        # Deduplicate by title
        seen = set()
        unique = []
        for h in all_headlines:
            if h["title"] not in seen:
                seen.add(h["title"])
                unique.append(h)
        unique = unique[:limit]

        # Classify
        if use_gemini:
            classified = await classify_headlines_batch(unique, symbol)
        else:
            classified = _keyword_classify(unique)

        # Compute score
        bull = sum(1 for h in classified if h.sentiment == "bullish")
        bear = sum(1 for h in classified if h.sentiment == "bearish")
        neutral = sum(1 for h in classified if h.sentiment == "neutral")
        total = len(classified) or 1

        # Weighted score: bullish=+1, neutral=0, bearish=-1, weighted by confidence
        weighted_sum = sum(
            (1.0 if h.sentiment == "bullish" else -1.0 if h.sentiment == "bearish" else 0.0) * h.confidence
            for h in classified
        )
        score = weighted_sum / total

        return SentimentResult(
            symbol=symbol,
            score=round(score, 3),
            bullish_count=bull,
            bearish_count=bear,
            neutral_count=neutral,
            headlines=classified,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )
