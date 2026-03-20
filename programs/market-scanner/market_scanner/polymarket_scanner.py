"""Polymarket scanner — find mispriced prediction markets.

Applies the same intelligence stack to Polymarket:
1. Fetch all active markets from Polymarket REST API
2. For each market: embed the question, match to recent news
3. Use Gemini to estimate "true" probability
4. Compare to market price — if delta > threshold, it's mispriced
5. Score by: delta * liquidity * news_recency

This is the prediction market equivalent of SMC pattern detection:
instead of finding order blocks, we're finding information asymmetries.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CLOB_REST_BASE = "https://clob.polymarket.com"
CURSOR_END = "LTE="


@dataclass
class PolymarketOpportunity:
    """A potentially mispriced prediction market."""
    condition_id: str
    question: str
    market_price_yes: float       # current YES price (0-1)
    estimated_prob: float          # our estimated true probability
    delta: float                   # estimated - market (positive = underpriced YES)
    liquidity_usd: float
    volume_usd: float
    end_date: str
    news_headlines: list[str]
    news_sentiment: float          # -1 to +1
    embedding_similarity: float    # how well news matches the question
    confidence: float              # 0-1 overall confidence
    direction: str                 # "buy_yes" | "buy_no" | "skip"
    reasoning: str


async def fetch_active_markets(limit: int = 100) -> list[dict]:
    """Fetch active Polymarket markets."""
    markets = []
    async with httpx.AsyncClient(timeout=30) as client:
        cursor = None
        while len(markets) < limit:
            params = {}
            if cursor:
                params["next_cursor"] = cursor
            try:
                resp = await client.get(f"{CLOB_REST_BASE}/markets", params=params)
                resp.raise_for_status()
                body = resp.json()
                for m in body.get("data", []):
                    if m.get("active") and not m.get("closed"):
                        tokens = m.get("tokens", [])
                        prices = {t.get("outcome", ""): float(t.get("price", 0) or 0) for t in tokens}
                        markets.append({
                            "condition_id": m.get("condition_id", ""),
                            "question": m.get("question", ""),
                            "price_yes": prices.get("Yes", 0.5),
                            "price_no": prices.get("No", 0.5),
                            "volume": float(m.get("volume", 0) or 0),
                            "liquidity": float(m.get("liquidity", 0) or 0),
                            "end_date": m.get("end_date_iso", ""),
                        })
                next_cursor = body.get("next_cursor", CURSOR_END)
                if next_cursor == CURSOR_END or not body.get("data"):
                    break
                cursor = next_cursor
            except Exception:
                break
    return markets[:limit]


async def estimate_probability(
    question: str,
    news_headlines: list[str],
    current_price: float,
) -> tuple[float, str]:
    """Use Gemini to estimate the true probability of a market question.

    Returns (estimated_probability, reasoning).
    """
    try:
        from oracle_shared.providers import get_llm
        llm = get_llm()

        news_text = "\n".join(f"- {h}" for h in news_headlines[:5]) if news_headlines else "No recent news found."

        prompt = (
            f"Prediction market question: \"{question}\"\n"
            f"Current market price (YES): {current_price:.1%}\n\n"
            f"Recent relevant news:\n{news_text}\n\n"
            f"Based on the news and your knowledge, estimate the TRUE probability "
            f"that this resolves YES. Return ONLY JSON:\n"
            f'{{"probability": 0.XX, "reasoning": "one sentence"}}'
        )

        result = await llm.generate_json(prompt, max_tokens=150)
        prob = float(result.get("probability", current_price))
        prob = max(0.01, min(0.99, prob))
        reasoning = result.get("reasoning", "")
        return prob, reasoning
    except Exception as e:
        logger.debug("Probability estimation failed: %s", e)
        return current_price, "Estimation failed — using market price"


async def scan_polymarket(
    min_delta: float = 0.08,
    min_liquidity: float = 1000,
    max_markets: int = 50,
) -> list[PolymarketOpportunity]:
    """Scan Polymarket for mispriced markets.

    Args:
        min_delta: minimum price vs estimated probability difference
        min_liquidity: minimum liquidity in USD
        max_markets: max markets to analyze (Gemini rate limit aware)
    """
    from market_scanner.news_sentiment import fetch_google_news
    from market_scanner.embedding_features import embed_text
    import numpy as np

    logger.info("Scanning Polymarket for opportunities...")
    markets = await fetch_active_markets(limit=200)
    logger.info("Fetched %d active markets", len(markets))

    # Filter by liquidity first
    liquid = [m for m in markets if m["liquidity"] >= min_liquidity]
    liquid.sort(key=lambda m: m["volume"], reverse=True)
    liquid = liquid[:max_markets]

    opportunities: list[PolymarketOpportunity] = []

    for market in liquid:
        question = market["question"]
        price_yes = market["price_yes"]

        # Fetch news related to this market question
        keywords = question[:80].replace("?", "").replace("Will ", "").replace("Is ", "")
        headlines_raw = await fetch_google_news(keywords, limit=5)
        headlines = [h["title"] for h in headlines_raw]

        # Compute news sentiment
        news_sentiment = 0.0
        if headlines:
            try:
                from market_scanner.news_sentiment import NewsSentiment
                ns = NewsSentiment()
                result = await ns.analyze(keywords[:30], use_gemini=False, limit=5)
                news_sentiment = result.score
            except Exception:
                pass

        # Embedding similarity between question and news
        embed_sim = 0.0
        try:
            q_vec = await embed_text(question)
            if q_vec and headlines:
                news_combined = ". ".join(headlines[:3])
                n_vec = await embed_text(news_combined)
                if n_vec:
                    q_arr = np.array(q_vec)
                    n_arr = np.array(n_vec)
                    embed_sim = float(np.dot(q_arr, n_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(n_arr) + 1e-8))
        except Exception:
            pass

        # Estimate true probability (uses Gemini — rate limited)
        estimated, reasoning = await estimate_probability(question, headlines, price_yes)
        delta = estimated - price_yes

        # Determine direction
        if delta > min_delta:
            direction = "buy_yes"
            confidence = min(1.0, abs(delta) * 3 + embed_sim * 0.3 + abs(news_sentiment) * 0.2)
        elif delta < -min_delta:
            direction = "buy_no"
            confidence = min(1.0, abs(delta) * 3 + embed_sim * 0.3 + abs(news_sentiment) * 0.2)
        else:
            direction = "skip"
            confidence = 0.0

        if direction != "skip":
            opportunities.append(PolymarketOpportunity(
                condition_id=market["condition_id"],
                question=question,
                market_price_yes=price_yes,
                estimated_prob=estimated,
                delta=round(delta, 3),
                liquidity_usd=market["liquidity"],
                volume_usd=market["volume"],
                end_date=market["end_date"],
                news_headlines=headlines,
                news_sentiment=round(news_sentiment, 3),
                embedding_similarity=round(embed_sim, 3),
                confidence=round(confidence, 3),
                direction=direction,
                reasoning=reasoning,
            ))

        # Rate limit for Gemini
        await asyncio.sleep(2)

    opportunities.sort(key=lambda o: abs(o.delta) * o.confidence, reverse=True)
    return opportunities
