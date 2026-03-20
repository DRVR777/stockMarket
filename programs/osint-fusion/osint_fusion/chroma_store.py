"""Steps 1 & 4 — ChromaDB setup and similarity search.

Step 1: Initialize PersistentClient, create oracle_markets collection,
        populate from oracle:state:markets Redis hash on startup.
Step 4: Query collection for each embedding, return matches above threshold.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import chromadb

from osint_fusion.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_DIMENSIONS,
    MARKET_STATE_KEY,
    SIMILARITY_N_RESULTS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages the oracle_markets ChromaDB collection."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaStore: initialized  collection=%s  persist=%s",
            CHROMA_COLLECTION_NAME,
            CHROMA_PERSIST_DIR,
        )

    async def populate_from_redis(
        self,
        redis_client: Any,
        embed_fn: Any,
    ) -> int:
        """Populate collection from oracle:state:markets Redis hash.

        ``embed_fn`` is an async callable: ``async def embed(texts: list[str]) -> list[list[float]]``

        Returns the number of markets upserted.
        """
        all_markets = await redis_client.hgetall(MARKET_STATE_KEY)
        if not all_markets:
            logger.info("ChromaStore: no markets in Redis — starting empty")
            return 0

        market_ids: list[str] = []
        questions: list[str] = []

        for market_id, raw_json in all_markets.items():
            try:
                data = json.loads(raw_json)
                question = data.get("market_question", "")
                if question:
                    market_ids.append(market_id)
                    questions.append(question)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("ChromaStore: skipping market %s — JSON parse error: %s", market_id, exc)
                continue

        if not questions:
            return 0

        # Embed market questions in batches of 100
        all_embeddings: list[list[float]] = []
        for i in range(0, len(questions), 100):
            batch = questions[i : i + 100]
            embeddings = await embed_fn(batch)
            all_embeddings.extend(embeddings)

        self._collection.upsert(
            ids=market_ids,
            embeddings=all_embeddings,
            documents=questions,
        )

        logger.info(
            "ChromaStore: populated %d markets from Redis", len(market_ids)
        )
        return len(market_ids)

    def upsert_market(
        self,
        market_id: str,
        question: str,
        embedding: list[float],
    ) -> None:
        """Add or update a single market in the collection."""
        self._collection.upsert(
            ids=[market_id],
            embeddings=[embedding],
            documents=[question],
        )

    def query(
        self,
        embedding: list[float],
        n_results: int = SIMILARITY_N_RESULTS,
    ) -> dict[str, float]:
        """Query for similar markets.

        Returns {market_id: similarity_score} for all matches above
        SIMILARITY_THRESHOLD. ChromaDB returns cosine distances, so
        similarity = 1 - distance.
        """
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches: dict[str, float] = {}
        for market_id, distance in zip(ids, distances):
            similarity = 1.0 - distance
            if similarity >= SIMILARITY_THRESHOLD:
                matches[market_id] = round(similarity, 4)

        return matches

    @property
    def count(self) -> int:
        return self._collection.count()
