"""Configuration for market-scanner."""
from __future__ import annotations

import os

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

# Scan intervals
CRYPTO_SCAN_INTERVAL: int = int(os.getenv("SCANNER_CRYPTO_INTERVAL", "300"))  # 5 min
STOCK_SCAN_INTERVAL: int = int(os.getenv("SCANNER_STOCK_INTERVAL", "900"))    # 15 min

# Top N assets to scan per provider
CRYPTO_TOP_N: int = int(os.getenv("SCANNER_CRYPTO_TOP_N", "200"))
STOCK_WATCHLIST: str = os.getenv(
    "SCANNER_STOCK_WATCHLIST",
    "SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,AMD,COIN,MARA,RIOT,MSTR,SQ,PYPL,SOFI,PLTR,ARM,SMCI"
)

# Pattern detection thresholds
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0
VOLUME_SPIKE_MULTIPLIER: float = 2.0   # 2x average volume
BB_SQUEEZE_PERCENTILE: float = 0.1      # bandwidth in bottom 10%
MIN_PATTERN_SCORE: float = 0.4          # minimum score to emit signal

# Redis channels
SCANNER_SIGNAL_CHANNEL: str = "oracle:scanner_signal"
SCANNER_STATE_KEY: str = "oracle:state:scanner"

LOG_LEVEL: str = os.getenv("SCANNER_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
