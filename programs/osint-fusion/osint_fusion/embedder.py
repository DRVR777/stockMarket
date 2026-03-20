"""Step 3 — Embedding pipeline.

Uses the pluggable provider abstraction — works with Gemini (free) or OpenAI.
Supports both single and batch embedding.
"""
from __future__ import annotations

import logging

from oracle_shared.providers import EmbeddingProvider, get_embedder

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps the embedding provider for OSFE."""

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider = provider or get_embedder()
        logger.info("Embedder: using %s (%d dims)", type(self._provider).__name__, self._provider.dimensions)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results = await self._provider.embed(texts)
        logger.debug("Embedder: embedded %d texts (%d dims)", len(texts), self._provider.dimensions)
        return results

    async def embed_single(self, text: str) -> list[float]:
        return await self._provider.embed_single(text)

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions
