"""Configuration constants for solana-executor (SOE)."""
from __future__ import annotations

import os

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Execution mode ───────────────────────────────────────────────────────────
PAPER_TRADING: bool = os.getenv("SOE_PAPER_TRADING", "true").lower() == "true"

# ── Redis key prefixes ───────────────────────────────────────────────────────
PARAMS_KEY: str = "oracle:state:params"
MODEL_KEY_PREFIX: str = "oracle:state:soe_model"
POSITIONS_KEY: str = "oracle:state:positions"
CIRCUIT_BREAKER_KEY: str = "oracle:state:circuit_breaker:soe"
DAILY_PNL_KEY: str = "oracle:state:daily_pnl:soe"

# ── Default trading params (overridden via Redis) ────────────────────────────
ENTRY_FLOOR_PCT: float = 0.05       # price within 5% of AI floor
TAKE_PROFIT_PCT: float = 0.08       # 8% above entry
STOP_LOSS_PCT: float = 0.04         # 4% below entry
MAX_POSITION_USD: float = 500.0
MAX_CONCURRENT_POSITIONS: int = 3
DAILY_LOSS_CEILING_USD: float = 200.0

# ── RE floor estimate ────────────────────────────────────────────────────────
FLOOR_REQUEST_CHANNEL: str = "oracle:re_floor_request"
FLOOR_RESPONSE_PREFIX: str = "oracle:re_floor_response"
FLOOR_ESTIMATE_INTERVAL_HOURS: int = 6

# ── Birdeye (Solana adapter) ─────────────────────────────────────────────────
BIRDEYE_REST_BASE: str = "https://public-api.birdeye.so"
BIRDEYE_WS_URL: str = "wss://public-api.birdeye.so/socket"

# ── Jupiter (Solana adapter) ─────────────────────────────────────────────────
JUPITER_QUOTE_URL: str = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL: str = "https://quote-api.jup.ag/v6/swap"
SLIPPAGE_BPS: int = 50  # 0.5%

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("SOE_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
