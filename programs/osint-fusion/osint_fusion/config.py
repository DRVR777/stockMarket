"""Configuration constants for osint-fusion (OSFE).

All values are read from environment variables with sensible defaults.
"""
from __future__ import annotations

import os

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHROMA_COLLECTION_NAME: str = "oracle_markets"
EMBEDDING_DIMENSIONS: int = 512

# ── OpenAI Embeddings ────────────────────────────────────────────────────────
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_BATCH_WAIT_MS: int = 500  # accumulate signals for this long before batch embed

# ── Anthropic (semantic state summary) ───────────────────────────────────────
HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
SUMMARY_MAX_WORDS: int = 100

# ── Similarity ───────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.65
SIMILARITY_N_RESULTS: int = 10

# ── MarketState ──────────────────────────────────────────────────────────────
# RECENT_INSIGHTS_WINDOW: use oracle_shared.contracts.market_state.RECENT_INSIGHTS_WINDOW
MARKET_STATE_KEY: str = "oracle:state:markets"

# ── Credibility weights ──────────────────────────────────────────────────────
CREDIBILITY_PARAMS_PREFIX: str = "oracle:state:params:credibility_weights"
CREDIBILITY_WEIGHT_MIN: float = 0.1
CREDIBILITY_WEIGHT_MAX: float = 2.0

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("OSFE_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
