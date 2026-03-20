"""Configuration for operator-dashboard."""
from __future__ import annotations

import os

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
PORT: int = int(os.getenv("DASHBOARD_PORT", "8080"))

# Redis keys
PARAMS_KEY: str = "oracle:state:params"

# All channels to subscribe to
ALL_CHANNELS: list[str] = [
    "oracle:signal",
    "oracle:anomaly_event",
    "oracle:insight",
    "oracle:market_state",
    "oracle:trade_thesis",
    "oracle:trade_execution",
    "oracle:post_mortem",
    "oracle:operator_alert",
]

# Telegram
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_RATE_LIMIT_SECONDS: float = 3.0

# Logging
LOG_LEVEL: str = os.getenv("DASHBOARD_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
