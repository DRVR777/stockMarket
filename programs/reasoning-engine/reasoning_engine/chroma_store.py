"""Task 1 — ChromaDB setup for oracle_theses collection.

Stores thesis embeddings for historical analogue search during context assembly.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import chromadb

from oracle_shared.contracts.trade_thesis import HistoricalAnalogue

from reasoning_engine.config import (
    ANALOGUE_N_RESULTS,
    ANALOGUE_SIMILARITY_THRESHOLD,
    CHROMA_PERSIST_DIR,
    CHROMA_THESES_COLLECTION,
)

logger = logging.getLogger(__name__)


class ThesisChromaStore:
    """Manages the oracle_theses ChromaDB collection for analogue search."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_THESES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ThesisChromaStore: initialized  collection=%s  count=%d",
            CHROMA_THESES_COLLECTION,
            self._collection.count(),
        )

    def upsert(
        self,
        thesis_id: str,
        market_question: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update a thesis embedding."""
        self._collection.upsert(
            ids=[thesis_id],
            embeddings=[embedding],
            documents=[market_question],
            metadatas=[metadata or {}],
        )

    def find_analogues(
        self,
        embedding: list[float],
        n_results: int = ANALOGUE_N_RESULTS,
    ) -> list[HistoricalAnalogue]:
        """Find historical thesis analogues by cosine similarity.

        Returns HistoricalAnalogue objects for matches above threshold.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self._collection.count()),
        )

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        analogues: list[HistoricalAnalogue] = []
        for thesis_id, distance, meta in zip(ids, distances, metadatas):
            similarity = 1.0 - distance
            if similarity >= ANALOGUE_SIMILARITY_THRESHOLD:
                analogues.append(HistoricalAnalogue(
                    thesis_id=thesis_id,
                    similarity=round(similarity, 4),
                    outcome=meta.get("outcome"),
                ))

        return analogues

    @property
    def count(self) -> int:
        return self._collection.count()
