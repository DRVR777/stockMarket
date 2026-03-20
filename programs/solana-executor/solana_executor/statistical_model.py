"""Task 2 — Per-asset statistical model.

Chain-agnostic: works with any asset that provides OHLCV data.
Maintains 30-day close prices, 20-day MA, standard deviation, velocity.
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from solana_executor.chains.base import OHLCVBar
from solana_executor.config import MODEL_KEY_PREFIX

logger = logging.getLogger(__name__)


@dataclass
class AssetModel:
    """Statistical model for a single asset."""
    token_address: str
    symbol: str
    chain: str
    prices_30d: list[float] = field(default_factory=list)
    ma_20: float = 0.0
    std_dev: float = 0.0
    price_velocity: float = 0.0
    current_price: float = 0.0
    ai_floor_estimate: float = 0.0
    ai_floor_estimated_at: str = ""

    def update_from_ohlcv(self, bars: list[OHLCVBar]) -> None:
        """Backfill from OHLCV bars."""
        self.prices_30d = [bar.close for bar in bars[-30:]]
        self._recompute()

    def update_price(self, price: float) -> None:
        """Update current price from a live tick."""
        self.current_price = price
        if len(self.prices_30d) > 0:
            oldest = self.prices_30d[0] if len(self.prices_30d) >= 2 else price
            self.price_velocity = (price - oldest) / oldest if oldest > 0 else 0.0

    def _recompute(self) -> None:
        if len(self.prices_30d) >= 20:
            self.ma_20 = statistics.mean(self.prices_30d[-20:])
        elif self.prices_30d:
            self.ma_20 = statistics.mean(self.prices_30d)

        if len(self.prices_30d) >= 2:
            self.std_dev = statistics.stdev(self.prices_30d)
        else:
            self.std_dev = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_address": self.token_address,
            "symbol": self.symbol,
            "chain": self.chain,
            "prices_30d": self.prices_30d,
            "ma_20": self.ma_20,
            "std_dev": self.std_dev,
            "price_velocity": self.price_velocity,
            "current_price": self.current_price,
            "ai_floor_estimate": self.ai_floor_estimate,
            "ai_floor_estimated_at": self.ai_floor_estimated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetModel:
        model = cls(
            token_address=data.get("token_address", ""),
            symbol=data.get("symbol", ""),
            chain=data.get("chain", ""),
        )
        model.prices_30d = data.get("prices_30d", [])
        model.ma_20 = data.get("ma_20", 0.0)
        model.std_dev = data.get("std_dev", 0.0)
        model.price_velocity = data.get("price_velocity", 0.0)
        model.current_price = data.get("current_price", 0.0)
        model.ai_floor_estimate = data.get("ai_floor_estimate", 0.0)
        model.ai_floor_estimated_at = data.get("ai_floor_estimated_at", "")
        return model


class ModelStore:
    """Persist AssetModel objects in Redis."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def save(self, model: AssetModel) -> None:
        key = f"{MODEL_KEY_PREFIX}:{model.token_address}"
        await self._redis.set(key, json.dumps(model.to_dict()))

    async def load(self, token_address: str) -> Optional[AssetModel]:
        key = f"{MODEL_KEY_PREFIX}:{token_address}"
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return AssetModel.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return None

    async def save_price(self, token_address: str, price: float) -> None:
        """Quick update of just the current price field."""
        model = await self.load(token_address)
        if model:
            model.update_price(price)
            await self.save(model)
