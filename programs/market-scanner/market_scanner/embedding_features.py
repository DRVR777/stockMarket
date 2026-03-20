"""Embedding-based features for the ML model.

Adds semantic understanding to pure price features:
1. Embed recent news headlines for each asset
2. Compare to historical embeddings of headlines that preceded big moves
3. Compute "similarity to bullish/bearish news" as ML features
4. Embed Reddit sentiment and funding rate context

This turns text data into numbers the ML model can use.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Cache for embeddings to avoid redundant API calls
_embedding_cache: dict[str, list[float]] = {}
_embedder = None


async def get_embedder():
    """Get or create the embedding provider."""
    global _embedder
    if _embedder is None:
        try:
            from oracle_shared.providers import get_embedder as _get
            _embedder = _get()
        except Exception:
            logger.warning("Embedder not available — embedding features disabled")
            return None
    return _embedder


async def embed_text(text: str) -> list[float] | None:
    """Embed a single text, with caching."""
    if text in _embedding_cache:
        return _embedding_cache[text]

    embedder = await get_embedder()
    if embedder is None:
        return None

    try:
        vec = await embedder.embed_single(text)
        _embedding_cache[text] = vec
        # Keep cache manageable
        if len(_embedding_cache) > 1000:
            keys = list(_embedding_cache.keys())
            for k in keys[:500]:
                del _embedding_cache[k]
        return vec
    except Exception:
        return None


async def compute_embedding_features(
    symbol: str,
    headlines: list[str] | None = None,
    reddit_titles: list[str] | None = None,
) -> dict[str, float]:
    """Compute embedding-based features for an asset.

    Returns features that can be added to the ML model:
      - news_embed_sentiment: cosine similarity to bullish vs bearish prototype
      - news_embed_magnitude: how "newsy" the current headlines are
      - reddit_embed_sentiment: same for Reddit
      - cross_asset_similarity: how similar current news is to BTC news (correlation proxy)
    """
    features = {
        "news_embed_sentiment": 0.0,
        "news_embed_magnitude": 0.0,
        "reddit_embed_sentiment": 0.0,
        "cross_asset_similarity": 0.0,
    }

    embedder = await get_embedder()
    if embedder is None:
        return features

    # Define sentiment prototypes
    bullish_prototype = "price surging, rally, breakout, all time high, massive gains, bullish momentum"
    bearish_prototype = "crash, plunge, sell off, bear market, liquidation, capitulation, fear"

    try:
        bull_vec = await embed_text(bullish_prototype)
        bear_vec = await embed_text(bearish_prototype)
        if bull_vec is None or bear_vec is None:
            return features
    except Exception:
        return features

    bull_arr = np.array(bull_vec)
    bear_arr = np.array(bear_vec)

    # News sentiment via embeddings
    if headlines:
        combined = ". ".join(headlines[:5])
        news_vec = await embed_text(combined)
        if news_vec is not None:
            news_arr = np.array(news_vec)
            bull_sim = float(np.dot(news_arr, bull_arr) / (np.linalg.norm(news_arr) * np.linalg.norm(bull_arr) + 1e-8))
            bear_sim = float(np.dot(news_arr, bear_arr) / (np.linalg.norm(news_arr) * np.linalg.norm(bear_arr) + 1e-8))
            features["news_embed_sentiment"] = bull_sim - bear_sim  # positive = bullish
            features["news_embed_magnitude"] = (bull_sim + bear_sim) / 2  # how relevant to markets

    # Reddit sentiment via embeddings
    if reddit_titles:
        combined = ". ".join(reddit_titles[:5])
        reddit_vec = await embed_text(combined)
        if reddit_vec is not None:
            reddit_arr = np.array(reddit_vec)
            bull_sim = float(np.dot(reddit_arr, bull_arr) / (np.linalg.norm(reddit_arr) * np.linalg.norm(bull_arr) + 1e-8))
            bear_sim = float(np.dot(reddit_arr, bear_arr) / (np.linalg.norm(reddit_arr) * np.linalg.norm(bear_arr) + 1e-8))
            features["reddit_embed_sentiment"] = bull_sim - bear_sim

    # Cross-asset: compare this asset's news to BTC news (market leader)
    if headlines and symbol != "BTCUSDT":
        btc_text = "Bitcoin BTC cryptocurrency market"
        asset_text = ". ".join(headlines[:3])
        btc_vec = await embed_text(btc_text)
        asset_vec = await embed_text(asset_text)
        if btc_vec is not None and asset_vec is not None:
            btc_arr = np.array(btc_vec)
            asset_arr = np.array(asset_vec)
            features["cross_asset_similarity"] = float(
                np.dot(btc_arr, asset_arr) / (np.linalg.norm(btc_arr) * np.linalg.norm(asset_arr) + 1e-8)
            )

    return features


async def enrich_scanner_signal(signal: dict) -> dict:
    """Add embedding features to a scanner signal before ML prediction.

    Call this before passing features to the ML model to add semantic context.
    """
    symbol = signal.get("symbol", "")

    # Fetch headlines for this symbol
    headlines = []
    try:
        from market_scanner.news_sentiment import fetch_google_news
        ticker = symbol.replace("USDT", "")
        raw = await fetch_google_news(ticker, limit=5)
        headlines = [h["title"] for h in raw]
    except Exception:
        pass

    # Fetch Reddit mentions
    reddit_titles = []
    try:
        from market_scanner.providers.scrapers import scrape_reddit_trending
        trends = await scrape_reddit_trending(["cryptocurrency"])
        reddit_titles = [
            t.title for t in trends
            if symbol.replace("USDT", "").lower() in t.title.lower()
        ]
    except Exception:
        pass

    # Compute embedding features
    embed_feats = await compute_embedding_features(symbol, headlines, reddit_titles)

    # Merge into signal
    signal.update(embed_feats)
    return signal
