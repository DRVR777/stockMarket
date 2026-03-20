"""Configuration constants for reasoning-engine (RE)."""
from __future__ import annotations

import os

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Postgres ─────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://oracle:oracle@localhost:5432/oracle"
)

# ── Anthropic ────────────────────────────────────────────────────────────────
SONNET_MODEL: str = "claude-sonnet-4-6"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHROMA_THESES_COLLECTION: str = "oracle_theses"
EMBEDDING_DIMENSIONS: int = 512
ANALOGUE_SIMILARITY_THRESHOLD: float = 0.7
ANALOGUE_N_RESULTS: int = 5

# ── OpenAI (for embedding theses into ChromaDB) ─────────────────────────────
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

# ── Pipeline thresholds ──────────────────────────────────────────────────────
# Minimum probability delta to act (default, overridden by Redis param)
RE_DELTA_THRESHOLD_DEFAULT: float = 0.10
RE_DELTA_THRESHOLD_PARAM: str = "re_delta_threshold"

# Minimum confidence to execute (default, overridden by Redis param)
RE_CONFIDENCE_MIN_DEFAULT: float = 0.45
RE_CONFIDENCE_MIN_PARAM: str = "re_confidence_min"

# Position sizing params
MAX_POSITION_USD_PARAM: str = "max_position_usd"
MAX_POSITION_USD_DEFAULT: float = 500.0
BASE_STAKE_USD_PARAM: str = "base_stake_usd"
BASE_STAKE_USD_DEFAULT: float = 1000.0

# ── Scheduler ────────────────────────────────────────────────────────────────
RE_SCAN_INTERVAL_PARAM: str = "re_scan_interval_minutes"
RE_SCAN_INTERVAL_DEFAULT: int = 30

# ── Redis keys ───────────────────────────────────────────────────────────────
PARAMS_KEY: str = "oracle:state:params"
MARKET_STATE_KEY: str = "oracle:state:markets"
RE_QUEUE_KEY: str = "oracle:state:re_queue"
THESES_INDEX_KEY: str = "oracle:state:theses_index"
ANOMALY_INDEX_PREFIX: str = "oracle:state:anomaly_index"

# SOE floor estimate channels
FLOOR_REQUEST_CHANNEL: str = "oracle:re_floor_request"
FLOOR_RESPONSE_PREFIX: str = "oracle:re_floor_response"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("RE_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
