"""Configuration constants for signal-ingestion (SIL).

All values are read from environment variables with sensible defaults.
No magic strings or numbers should appear in adapter code — import from here.
"""
from __future__ import annotations

import os

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_DEFAULT_URL = "redis://localhost:6379"
REDIS_URL: str = os.getenv("REDIS_URL", REDIS_DEFAULT_URL)

# ── Adapter polling intervals (seconds) ─────────────────────────────────────
POLYMARKET_REST_POLL_INTERVAL: int = int(os.getenv("POLYMARKET_REST_POLL_INTERVAL", "60"))
POLYMARKET_WS_MAX_MARKETS: int = int(os.getenv("POLYMARKET_WS_MAX_MARKETS", "200"))
POLYMARKET_WS_RECONNECT_DELAY: float = float(os.getenv("POLYMARKET_WS_RECONNECT_DELAY", "5.0"))

POLYGON_RECONNECT_DELAY: float = float(os.getenv("POLYGON_RECONNECT_DELAY", "5.0"))

NEWSAPI_POLL_INTERVAL: int = int(os.getenv("NEWSAPI_POLL_INTERVAL", "300"))
WIKIPEDIA_POLL_INTERVAL: int = int(os.getenv("WIKIPEDIA_POLL_INTERVAL", "900"))
REDDIT_POLL_INTERVAL: int = int(os.getenv("REDDIT_POLL_INTERVAL", "600"))

BIRDEYE_RECONNECT_DELAY: float = float(os.getenv("BIRDEYE_RECONNECT_DELAY", "5.0"))
BIRDEYE_REST_FALLBACK_INTERVAL: float = float(os.getenv("BIRDEYE_REST_FALLBACK_INTERVAL", "5.0"))

AI_OPINION_POLL_INTERVAL: int = int(os.getenv("AI_OPINION_POLL_INTERVAL", "1800"))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("SIL_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
