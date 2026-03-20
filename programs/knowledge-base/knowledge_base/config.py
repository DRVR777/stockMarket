"""Configuration constants for knowledge-base (KBPM)."""
from __future__ import annotations

import os

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Vault ────────────────────────────────────────────────────────────────────
VAULT_DIR: str = os.getenv("KBPM_VAULT_DIR", "./data/vault")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHROMA_THESES_COLLECTION: str = "oracle_theses"
EMBEDDING_DIMENSIONS: int = 512
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

# ── Anthropic ────────────────────────────────────────────────────────────────
SONNET_MODEL: str = "claude-sonnet-4-6"

# ── Redis keys ───────────────────────────────────────────────────────────────
THESES_INDEX_KEY: str = "oracle:state:theses_index"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("KBPM_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
