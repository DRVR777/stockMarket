"""Tasks 2 & 3 — ThesisWriter and ThesisIndexer.

Subscribes to oracle:trade_thesis. Writes vault/theses/{thesis_id}.md.
Indexes each thesis in ChromaDB for RE analogue search.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import chromadb
from oracle_shared.contracts.trade_thesis import TradeThesis
from oracle_shared.db import get_session
from oracle_shared.db.repository import ThesisRepo
from oracle_shared.providers import EmbeddingProvider, get_embedder

from knowledge_base.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_THESES_COLLECTION,
    THESES_INDEX_KEY,
)
from knowledge_base.vault import vault_path, write_md

logger = logging.getLogger(__name__)


class ThesisWriter:
    """Subscribe to TradeThesis, write to vault, and index in ChromaDB."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._running = False
        self._chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_THESES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embedder()

    async def start(self) -> None:
        self._running = True
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(TradeThesis.CHANNEL)
        logger.info("ThesisWriter: subscribed to %s", TradeThesis.CHANNEL)

        try:
            while self._running:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if msg is None:
                    await asyncio.sleep(0.05)
                    continue
                if msg["type"] != "message":
                    continue

                try:
                    thesis = TradeThesis.model_validate_json(msg["data"])
                    await self._process(thesis)
                except Exception:
                    logger.warning("ThesisWriter: failed to process", exc_info=True)
        finally:
            await pubsub.unsubscribe(TradeThesis.CHANNEL)
            await pubsub.aclose()

    async def _process(self, thesis: TradeThesis) -> None:
        """Task 2: Write thesis file. Task 3: Index in ChromaDB."""
        # Task 2: Write vault/theses/{thesis_id}.md (never overwrite)
        path = vault_path("theses", f"{thesis.thesis_id}.md")
        front_matter = {
            "thesis_id": thesis.thesis_id,
            "market_id": thesis.market_id,
            "decision": thesis.decision.value,
            "confidence_score": thesis.confidence_score,
            "direction": thesis.direction,
            "outcome": None,
        }
        body = f"# {thesis.market_question}\n\n```json\n{thesis.model_dump_json(indent=2)}\n```"
        written = write_md(path, front_matter, body, overwrite=False)

        if written:
            logger.info("ThesisWriter: wrote %s", path.name)

        # Task 3: Index in ChromaDB
        try:
            embed_text = (
                f"{thesis.market_question} {thesis.direction} "
                f"{thesis.hypotheses[0].argument if thesis.hypotheses else ''}"
            )
            embedding = await self._embedder.embed_single(embed_text)
            self._collection.upsert(
                ids=[thesis.thesis_id],
                embeddings=[embedding],
                documents=[thesis.market_question],
                metadatas=[{
                    "market_id": thesis.market_id,
                    "decision": thesis.decision.value,
                    "confidence_score": str(thesis.confidence_score),
                    "outcome": "",
                }],
            )
            logger.debug("ThesisWriter: indexed %s in ChromaDB", thesis.thesis_id)
        except Exception:
            logger.warning("ThesisWriter: ChromaDB index failed", exc_info=True)

        # Index thesis_id in Redis set
        await self._redis.sadd(THESES_INDEX_KEY, thesis.thesis_id)

    async def stop(self) -> None:
        self._running = False
        logger.info("ThesisWriter: stopped")
