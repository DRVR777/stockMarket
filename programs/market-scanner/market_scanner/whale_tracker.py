"""Whale wallet tracker — paper trade whale moves, build a smart money rolodex.

Monitors on-chain whale activity and:
1. Paper trades every whale move (buy when they buy, sell when they sell)
2. Tracks 24h/7d outcome of each whale's trades
3. Builds a wallet reputation database: which wallets are consistently profitable?
4. When a proven profitable whale moves, that signal gets boosted confidence

Over time, this creates a "smart money filter":
  - New unknown whale buys $50k → moderate signal
  - Whale with 72% historical win rate buys $50k → high confidence signal

Storage: Redis for real-time tracking, Postgres for permanent history
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

WHALE_TRACKER_KEY = "oracle:whale_tracker"
WHALE_ROSTER_KEY = "oracle:whale_roster"
WHALE_TRADES_KEY = "oracle:whale_trades"


@dataclass
class WhaleWallet:
    """Tracked whale wallet with performance history."""
    address: str
    first_seen: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_trade_size_usd: float = 0.0
    best_trade_pnl_pct: float = 0.0
    worst_trade_pnl_pct: float = 0.0
    last_active: str = ""
    tier: str = "unknown"  # "shark", "informed", "noise", "unknown"
    tags: list[str] = field(default_factory=list)  # "consistent", "big_mover", "dip_buyer", etc.


@dataclass
class WhaleTrade:
    """A single tracked whale trade with outcome."""
    wallet: str
    symbol: str
    direction: str          # "buy" | "sell"
    size_usd: float
    entry_price: float
    entry_time: str
    # Outcome (filled after 24h)
    price_24h: float | None = None
    pnl_pct: float | None = None
    outcome: str | None = None  # "win" | "loss" | "pending"


class WhaleTracker:
    """Track whale wallet performance and build smart money rolodex."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._pending_trades: list[WhaleTrade] = []

    async def on_whale_signal(
        self,
        wallet_address: str,
        symbol: str,
        direction: str,
        size_usd: float,
        price: float,
    ) -> dict:
        """Record a new whale trade and return wallet reputation.

        Called when the system detects a large on-chain transaction.
        Returns the wallet's historical performance for confidence scoring.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Record the trade
        trade = WhaleTrade(
            wallet=wallet_address,
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            entry_price=price,
            entry_time=now,
            outcome="pending",
        )
        self._pending_trades.append(trade)

        # Save to Redis
        trade_data = {
            "wallet": wallet_address,
            "symbol": symbol,
            "direction": direction,
            "size_usd": size_usd,
            "entry_price": price,
            "entry_time": now,
            "outcome": "pending",
        }
        await self._redis.lpush(WHALE_TRADES_KEY, json.dumps(trade_data))
        await self._redis.ltrim(WHALE_TRADES_KEY, 0, 9999)

        # Get or create wallet profile
        wallet = await self._get_wallet(wallet_address)
        if wallet is None:
            wallet = WhaleWallet(
                address=wallet_address,
                first_seen=now,
                last_active=now,
            )
            await self._save_wallet(wallet)

        wallet.last_active = now

        logger.info(
            "WhaleTracker: %s %s $%.0f %s  (wallet WR=%.0f%% over %d trades, tier=%s)",
            direction.upper(), symbol, size_usd, wallet_address[:12],
            wallet.win_rate * 100, wallet.total_trades, wallet.tier,
        )

        return {
            "wallet": wallet_address,
            "win_rate": wallet.win_rate,
            "total_trades": wallet.total_trades,
            "tier": wallet.tier,
            "avg_size": wallet.avg_trade_size_usd,
            "total_pnl": wallet.total_pnl_pct,
        }

    async def check_outcomes(self) -> list[WhaleTrade]:
        """Check 24h outcomes for pending whale trades."""
        from market_scanner.providers.binance_ws import BinanceKlineProvider
        provider = BinanceKlineProvider()

        resolved = []
        now = datetime.now(timezone.utc)

        for trade in self._pending_trades[:]:
            entry_time = datetime.fromisoformat(trade.entry_time)
            if now - entry_time < timedelta(hours=24):
                continue

            try:
                candles = await provider.get_recent_klines(trade.symbol, interval="1h", limit=1)
                if not candles:
                    continue

                current_price = candles[0].close
                if trade.direction == "buy":
                    pnl = (current_price - trade.entry_price) / trade.entry_price * 100
                else:
                    pnl = (trade.entry_price - current_price) / trade.entry_price * 100

                trade.price_24h = current_price
                trade.pnl_pct = round(pnl, 3)
                trade.outcome = "win" if pnl > 0 else "loss"

                # Update wallet stats
                await self._update_wallet_stats(trade)
                resolved.append(trade)
                self._pending_trades.remove(trade)

                logger.info(
                    "WhaleTracker outcome: %s %s -> %s  pnl=%+.2f%%",
                    trade.wallet[:12], trade.symbol, trade.outcome, pnl,
                )
            except Exception:
                pass

        return resolved

    async def get_roster(self, min_trades: int = 5) -> list[WhaleWallet]:
        """Get all tracked wallets sorted by win rate."""
        wallets = []
        roster_data = await self._redis.hgetall(WHALE_ROSTER_KEY)

        for addr, raw in roster_data.items():
            try:
                data = json.loads(raw)
                wallet = WhaleWallet(**data)
                if wallet.total_trades >= min_trades:
                    wallets.append(wallet)
            except Exception:
                pass

        wallets.sort(key=lambda w: (w.win_rate, w.total_trades), reverse=True)
        return wallets

    async def get_wallet_confidence(self, wallet_address: str) -> float:
        """Get confidence multiplier for a wallet based on history.

        Returns:
          1.5 for proven sharks (>65% WR, 10+ trades)
          1.2 for informed wallets (>55% WR, 5+ trades)
          1.0 for unknown wallets
          0.7 for known noise wallets (<40% WR)
        """
        wallet = await self._get_wallet(wallet_address)
        if wallet is None or wallet.total_trades < 3:
            return 1.0

        if wallet.total_trades >= 10 and wallet.win_rate >= 0.65:
            return 1.5
        elif wallet.total_trades >= 5 and wallet.win_rate >= 0.55:
            return 1.2
        elif wallet.total_trades >= 5 and wallet.win_rate < 0.40:
            return 0.7
        return 1.0

    async def _update_wallet_stats(self, trade: WhaleTrade) -> None:
        """Update wallet statistics after a trade resolves."""
        wallet = await self._get_wallet(trade.wallet)
        if wallet is None:
            wallet = WhaleWallet(
                address=trade.wallet,
                first_seen=trade.entry_time,
            )

        wallet.total_trades += 1
        if trade.outcome == "win":
            wallet.wins += 1
        else:
            wallet.losses += 1

        wallet.win_rate = wallet.wins / wallet.total_trades if wallet.total_trades > 0 else 0
        wallet.total_pnl_pct += (trade.pnl_pct or 0)

        # Rolling average trade size
        wallet.avg_trade_size_usd = (
            (wallet.avg_trade_size_usd * (wallet.total_trades - 1) + trade.size_usd)
            / wallet.total_trades
        )

        if trade.pnl_pct and trade.pnl_pct > wallet.best_trade_pnl_pct:
            wallet.best_trade_pnl_pct = trade.pnl_pct
        if trade.pnl_pct and trade.pnl_pct < wallet.worst_trade_pnl_pct:
            wallet.worst_trade_pnl_pct = trade.pnl_pct

        wallet.last_active = datetime.now(timezone.utc).isoformat()

        # Assign tier
        if wallet.total_trades >= 10 and wallet.win_rate >= 0.65:
            wallet.tier = "shark"
        elif wallet.total_trades >= 5 and wallet.win_rate >= 0.55:
            wallet.tier = "informed"
        elif wallet.total_trades >= 5 and wallet.win_rate < 0.40:
            wallet.tier = "noise"
        else:
            wallet.tier = "unknown"

        # Tags
        wallet.tags = []
        if wallet.win_rate >= 0.7 and wallet.total_trades >= 10:
            wallet.tags.append("consistent")
        if wallet.avg_trade_size_usd >= 50000:
            wallet.tags.append("big_mover")
        if wallet.total_pnl_pct > 50:
            wallet.tags.append("highly_profitable")

        await self._save_wallet(wallet)

    async def _get_wallet(self, address: str) -> WhaleWallet | None:
        raw = await self._redis.hget(WHALE_ROSTER_KEY, address)
        if raw is None:
            return None
        try:
            return WhaleWallet(**json.loads(raw))
        except Exception:
            return None

    async def _save_wallet(self, wallet: WhaleWallet) -> None:
        data = {
            "address": wallet.address,
            "first_seen": wallet.first_seen,
            "total_trades": wallet.total_trades,
            "wins": wallet.wins,
            "losses": wallet.losses,
            "win_rate": round(wallet.win_rate, 3),
            "total_pnl_pct": round(wallet.total_pnl_pct, 3),
            "avg_trade_size_usd": round(wallet.avg_trade_size_usd, 2),
            "best_trade_pnl_pct": round(wallet.best_trade_pnl_pct, 3),
            "worst_trade_pnl_pct": round(wallet.worst_trade_pnl_pct, 3),
            "last_active": wallet.last_active,
            "tier": wallet.tier,
            "tags": wallet.tags,
        }
        await self._redis.hset(WHALE_ROSTER_KEY, wallet.address, json.dumps(data))
