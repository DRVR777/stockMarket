"""Configuration constants for whale-detector (WADE).

All values are read from environment variables with sensible defaults.
No magic strings or numbers should appear in pipeline code — import from here.
"""
from __future__ import annotations

import os

# -- Redis -------------------------------------------------------------------
REDIS_DEFAULT_URL = "redis://localhost:6379"
REDIS_URL: str = os.getenv("REDIS_URL", REDIS_DEFAULT_URL)

# -- Threshold parameters (Step 2) ------------------------------------------
# Default large-order threshold in USD. Overridden at runtime by reading
# oracle:state:params:large_order_threshold_usd from Redis.
LARGE_ORDER_THRESHOLD_USD_DEFAULT: float = 5_000.0

# Redis key for the operator-configurable large-order threshold
LARGE_ORDER_THRESHOLD_PARAM_KEY: str = "large_order_threshold_usd"

# -- Copy-trade eligibility (Step 6) ----------------------------------------
# Default copy-trade anomaly-score threshold. Overridden at runtime by
# oracle:state:params:copy_trade_threshold from Redis.
COPY_TRADE_THRESHOLD_DEFAULT: float = 0.7

COPY_TRADE_THRESHOLD_PARAM_KEY: str = "copy_trade_threshold"

# -- Anomaly scoring (Step 4) -----------------------------------------------
# Equal weight for three scoring factors (size_ratio, wallet_ratio, time_ratio)
ANOMALY_WEIGHT_SIZE: float = 1.0 / 3.0
ANOMALY_WEIGHT_WALLET: float = 1.0 / 3.0
ANOMALY_WEIGHT_TIME: float = 1.0 / 3.0

# Wallet position-size ratio threshold: if size >= N * typical, score = 1.0
WALLET_POSITION_MULTIPLIER_CAP: float = 2.0

# Time-to-resolution horizon in hours (7 days)
RESOLUTION_HORIZON_HOURS: float = 168.0

# -- Cascade detection (Step 5) ---------------------------------------------
# Time window in seconds to look back for coordinated activity
CASCADE_WINDOW_SECONDS: int = 300

# Minimum distinct wallets to trigger cascade flag
CASCADE_MIN_WALLETS: int = 3

# TTL for cascade sorted sets in seconds
CASCADE_SET_TTL_SECONDS: int = 600

# -- Wallet registry (Steps 3 & 8) ------------------------------------------
# Maximum fills stored per wallet for rolling median calculation
WALLET_FILLS_MAX_LENGTH: int = 20

# Redis key prefix for per-wallet fill lists
WALLET_FILLS_KEY_PREFIX: str = "oracle:state:wallet_fills"

# Redis key for operator-configurable params hash
PARAMS_STATE_KEY: str = "oracle:state:params"

# Redis key prefix for market state
MARKET_STATE_KEY: str = "oracle:state:markets"

# Redis key prefix for cascade sorted sets
CASCADE_KEY_PREFIX: str = "oracle:state:cascade"

# -- Logging -----------------------------------------------------------------
LOG_LEVEL: str = os.getenv("WADE_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
