"""ORACLE Daemon — always-on pipeline that runs everything.

Starts 4 concurrent processes:
  1. Live Scanner: streams Binance, detects patterns, filters with ML
  2. Trade Journal: logs every signal to Postgres with outcome tracking
  3. Paper Trader: simulates executing trades, tracks PnL
  4. Auto-Retrain: weekly model retraining on new data

Run: python -m market_scanner.daemon
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Trade Journal ─────────────────────────────────────────────────────────────

class TradeJournal:
    """Log every signal to Redis + Postgres with 24h outcome tracking."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._pending: list[dict] = []  # signals awaiting outcome

    async def log_signal(self, signal: dict) -> None:
        """Log a new signal. Will check outcome later."""
        entry = {
            "signal_id": signal.get("symbol", "") + "_" + str(int(time.time())),
            "symbol": signal.get("symbol", ""),
            "direction": signal.get("direction", ""),
            "entry_price": signal.get("current_price", 0),
            "ml_probability": signal.get("ml_win_probability", signal.get("smc_confidence", 0)),
            "setup_type": signal.get("setup_type", ""),
            "signals": signal.get("signals", []),
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "check_at": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
            "outcome": None,
            "outcome_price": None,
            "pnl_pct": None,
        }
        self._pending.append(entry)

        # Store in Redis
        await self._redis.lpush("oracle:trade_journal", json.dumps(entry, default=str))
        await self._redis.ltrim("oracle:trade_journal", 0, 9999)

        # Persist to Postgres
        try:
            from oracle_shared.db import get_session
            from oracle_shared.db.models import SignalRow
            from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId
            sig = Signal(
                source_id=SourceId.AI_OPINION,
                timestamp=datetime.now(timezone.utc),
                category=SignalCategory.PRICE,
                raw_payload=entry,
                confidence=entry["ml_probability"],
            )
            async with get_session() as session:
                session.add(SignalRow.from_contract(sig))
        except Exception:
            pass

        logger.info(
            "Journal: %s %s %s  ml=%.0f%%  price=$%.2f",
            entry["direction"], entry["symbol"], entry["setup_type"],
            entry["ml_probability"] * 100, entry["entry_price"],
        )

    async def check_outcomes(self) -> None:
        """Check 24h outcomes for pending signals."""
        from market_scanner.providers.binance_ws import BinanceKlineProvider
        provider = BinanceKlineProvider()
        now = datetime.now(timezone.utc)
        resolved = []

        for entry in self._pending:
            check_time = datetime.fromisoformat(entry["check_at"])
            if now < check_time:
                continue

            try:
                candles = await provider.get_recent_klines(entry["symbol"], interval="1h", limit=1)
                if candles:
                    current_price = candles[0].close
                    entry_price = entry["entry_price"]
                    if entry["direction"] == "long":
                        pnl = (current_price - entry_price) / entry_price * 100
                    else:
                        pnl = (entry_price - current_price) / entry_price * 100

                    entry["outcome"] = "win" if pnl > 0 else "loss"
                    entry["outcome_price"] = current_price
                    entry["pnl_pct"] = round(pnl, 3)
                    resolved.append(entry)

                    logger.info(
                        "Journal outcome: %s %s -> %s  pnl=%+.2f%%",
                        entry["symbol"], entry["direction"], entry["outcome"], pnl,
                    )
            except Exception:
                pass

        for entry in resolved:
            self._pending.remove(entry)
            await self._redis.lpush("oracle:journal_resolved", json.dumps(entry, default=str))

    def get_stats(self) -> dict:
        """Return current journal stats."""
        return {
            "pending": len(self._pending),
            "total_logged": 0,  # would query Postgres
        }


# ── Paper Trader ──────────────────────────────────────────────────────────────

class PaperTrader:
    """Simulate executing trades based on scanner signals."""

    def __init__(
        self,
        redis_client: Any,
        capital: float = 10_000,
        risk_per_trade_pct: float = 2.0,
        max_concurrent: int = 5,
    ) -> None:
        self._redis = redis_client
        self._capital = capital
        self._starting = capital
        self._risk_pct = risk_per_trade_pct
        self._max_concurrent = max_concurrent
        self._open_trades: list[dict] = []
        self._closed_trades: list[dict] = []
        self._daily_pnl = 0.0

    async def on_signal(self, signal: dict) -> None:
        """Evaluate a signal for paper trading."""
        if len(self._open_trades) >= self._max_concurrent:
            return

        ml_prob = signal.get("ml_win_probability", signal.get("smc_confidence", 0))
        if ml_prob < 0.55:
            return

        pos_size = self._capital * self._risk_pct / 100
        trade = {
            "symbol": signal.get("symbol", ""),
            "direction": signal.get("direction", ""),
            "entry_price": signal.get("current_price", 0),
            "size_usd": pos_size,
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "ml_prob": ml_prob,
        }
        self._open_trades.append(trade)
        logger.info(
            "Paper: OPEN %s %s $%.0f @ $%.2f  ml=%.0f%%",
            trade["direction"], trade["symbol"], pos_size, trade["entry_price"], ml_prob * 100,
        )

    async def check_positions(self) -> None:
        """Check open positions for TP/SL hits."""
        from market_scanner.providers.binance_ws import BinanceKlineProvider
        provider = BinanceKlineProvider()
        to_close = []

        for trade in self._open_trades:
            try:
                candles = await provider.get_recent_klines(trade["symbol"], interval="1m", limit=1)
                if not candles:
                    continue
                price = candles[0].close

                close_reason = None
                sl = trade.get("stop_loss")
                tp = trade.get("take_profit")

                if trade["direction"] == "long":
                    if tp is not None and price >= tp:
                        close_reason = "take_profit"
                    elif sl is not None and price <= sl:
                        close_reason = "stop_loss"
                else:
                    if tp is not None and price <= tp:
                        close_reason = "take_profit"
                    elif sl is not None and price >= sl:
                        close_reason = "stop_loss"

                # Timeout: close after 24h if no SL/TP hit
                from dateutil import parser as dp
                opened = dp.parse(trade.get("opened_at", datetime.now(timezone.utc).isoformat()))
                if (datetime.now(timezone.utc) - opened.replace(tzinfo=timezone.utc if opened.tzinfo is None else opened.tzinfo)) > timedelta(hours=24):
                    if not close_reason:
                        close_reason = "timeout"

                if close_reason:
                    if trade["direction"] == "long":
                        pnl_pct = (price - trade["entry_price"]) / trade["entry_price"] * 100
                    else:
                        pnl_pct = (trade["entry_price"] - price) / trade["entry_price"] * 100

                    pnl_usd = trade["size_usd"] * pnl_pct / 100
                    self._capital += pnl_usd
                    self._daily_pnl += pnl_usd

                    trade["exit_price"] = price
                    trade["pnl_pct"] = round(pnl_pct, 3)
                    trade["pnl_usd"] = round(pnl_usd, 2)
                    trade["close_reason"] = close_reason
                    trade["closed_at"] = datetime.now(timezone.utc).isoformat()

                    to_close.append(trade)
                    self._closed_trades.append(trade)

                    logger.info(
                        "Paper: CLOSE %s %s  reason=%s  pnl=$%.2f (%.2f%%)  capital=$%.0f",
                        trade["symbol"], trade["direction"], close_reason, pnl_usd, pnl_pct, self._capital,
                    )
            except Exception:
                pass

        for trade in to_close:
            self._open_trades.remove(trade)

    def get_stats(self) -> dict:
        wins = sum(1 for t in self._closed_trades if t.get("pnl_usd", 0) > 0)
        total = len(self._closed_trades)
        return {
            "capital": round(self._capital, 2),
            "return_pct": round((self._capital - self._starting) / self._starting * 100, 2),
            "open_trades": len(self._open_trades),
            "closed_trades": total,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "daily_pnl": round(self._daily_pnl, 2),
        }


# ── Daemon ────────────────────────────────────────────────────────────────────

class OracleDaemon:
    """Main daemon that orchestrates all processes."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._journal = TradeJournal(redis_client)
        self._paper = PaperTrader(redis_client)
        self._running = False

        # Data pipeline — persists everything to Postgres + embeds text
        from market_scanner.data_pipeline import DataPipeline
        self._pipeline = DataPipeline(redis_client)

        # Knowledge graph — continuous semantic memory in ChromaDB
        from market_scanner.knowledge_graph import KnowledgeGraph
        self._kg = KnowledgeGraph()

    async def start(self) -> None:
        """Start all processes concurrently."""
        self._running = True
        logger.info("ORACLE Daemon starting...")

        await asyncio.gather(
            self._run_scanner(),
            self._run_position_checker(),
            self._run_outcome_checker(),
            self._run_status_reporter(),
            self._run_stock_scanner(),
            self._run_funding_scanner(),
            self._run_polymarket_scanner(),
            self._run_cascade_monitor(),
            self._run_kg_market_snapshots(),
        )

    async def _run_scanner(self) -> None:
        """Run the live scanner and feed signals to journal + paper trader."""
        from market_scanner.live_scanner import LiveScanner

        scanner = LiveScanner(
            self._redis,
            symbols=[
                # Top 30 by volume — 3x more signal surface area
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "APTUSDT", "NEARUSDT",
                "ADAUSDT", "DOTUSDT", "UNIUSDT", "AAVEUSDT", "LTCUSDT",
                "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT", "FETUSDT",
                "RENDERUSDT", "TIAUSDT", "LDOUSDT", "ATOMUSDT", "MKRUSDT",
                "FILUSDT", "STXUSDT", "IMXUSDT", "RUNEUSDT", "PENDLEUSDT",
            ],
            interval="15m",
            ml_threshold=0.5,
        )

        # Override the scanner's publish to also feed our journal + paper trader
        original_publish = scanner._publish

        async def augmented_publish(result):
            await original_publish(result)
            signal_data = {
                "symbol": result.symbol,
                "direction": result.direction,
                "current_price": result.current_price,
                "ml_win_probability": result.ml_probability,
                "smc_confidence": result.confidence,
                "setup_type": result.setup_type,
                "signals": result.signals,
                "stop_loss": result.stop_loss,      # actual SMC support/resistance level
                "take_profit": result.take_profit,  # actual SMC target level
            }
            await self._pipeline.process_signal(signal_data)
            await self._journal.log_signal(signal_data)
            await self._paper.on_signal(signal_data)
            await self._kg.add_signal(signal_data)

        scanner._publish = augmented_publish
        await scanner.start()

    async def _run_position_checker(self) -> None:
        """Check paper positions every minute."""
        while self._running:
            await self._paper.check_positions()
            await asyncio.sleep(60)

    async def _run_outcome_checker(self) -> None:
        """Check signal outcomes every hour."""
        while self._running:
            await self._journal.check_outcomes()
            await asyncio.sleep(3600)

    async def _run_status_reporter(self) -> None:
        """Print status every 5 minutes."""
        while self._running:
            await asyncio.sleep(300)
            paper = self._paper.get_stats()
            journal = self._journal.get_stats()
            kg_stats = self._kg.stats()
            logger.info(
                "STATUS: capital=$%.0f (%+.1f%%)  open=%d  closed=%d  wr=%.0f%%  "
                "journal=%d  kg_signals=%d  kg_outcomes=%d  kg_snapshots=%d",
                paper["capital"], paper["return_pct"], paper["open_trades"],
                paper["closed_trades"], paper["win_rate"], journal["pending"],
                kg_stats["signals"], kg_stats["outcomes"], kg_stats["market_snapshots"],
            )
            # Save to Redis for dashboard
            await self._redis.set("oracle:daemon_status", json.dumps({
                "paper": paper,
                "journal": journal,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }))

    async def _run_stock_scanner(self) -> None:
        """Scan stocks every 15 minutes during market hours (9:30am-4pm ET)."""
        from market_scanner.providers.stocks import StockProvider
        from market_scanner.smc import analyze_smc
        import pandas as pd

        stocks = StockProvider()
        tickers = [
            "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "SPY", "QQQ", "COIN",
            "MSTR", "PLTR", "AMZN", "META", "SMCI", "ARM", "SOFI",
            "MARA", "RIOT", "NFLX", "CRM", "UBER",
        ]

        while self._running:
            try:
                for ticker in tickers:
                    df = await stocks.get_ohlcv(ticker, period="3mo")
                    if len(df) < 60:
                        continue
                    smc = analyze_smc(df, ticker, "stock")
                    if smc and smc.confidence >= 0.6 and smc.setup_type in ("ob_retest", "fvg_fill"):
                        signal_data = {
                            "symbol": ticker,
                            "direction": "long" if smc.bias.value == "bullish" else "short",
                            "current_price": smc.current_price,
                            "ml_win_probability": smc.confidence,
                            "smc_confidence": smc.confidence,
                            "setup_type": smc.setup_type,
                            "signals": smc.signals[:5],
                            "stop_loss": smc.stop_loss,
                            "take_profit": smc.take_profit,
                            "asset_type": "stock",
                        }
                        await self._pipeline.process_signal(signal_data)
                        await self._journal.log_signal(signal_data)
                        await self._paper.on_signal(signal_data)
                        await self._kg.add_signal(signal_data)
                        logger.info("Stock signal: %s %s  conf=%.0f%%  setup=%s",
                                    signal_data["direction"].upper(), ticker,
                                    smc.confidence * 100, smc.setup_type)
            except Exception:
                logger.debug("Stock scan cycle failed", exc_info=True)

            await asyncio.sleep(900)  # every 15 min

    async def _run_funding_scanner(self) -> None:
        """Scan funding rates every hour for carry trade opportunities."""
        from market_scanner.funding_scanner import scan_funding

        while self._running:
            try:
                opps = await scan_funding(min_rate_pct=0.05, min_volume_usd=5_000_000)
                for opp in opps[:5]:
                    signal_data = {
                        "symbol": opp.symbol,
                        "direction": opp.direction,
                        "current_price": opp.price,
                        "ml_win_probability": 0.7,  # funding farming has inherent edge
                        "smc_confidence": 0.0,
                        "setup_type": "funding_farm",
                        "signals": [f"rate={opp.funding_rate:+.3f}%", f"daily={opp.daily_income_pct:.2f}%"],
                        "stop_loss": None,
                        "take_profit": None,
                        "asset_type": "crypto_funding",
                    }
                    await self._pipeline.process_signal(signal_data)
                    await self._journal.log_signal(signal_data)
                    await self._kg.add_signal(signal_data)
                    logger.info("Funding signal: %s %s  rate=%+.3f%%/8h  daily=%.2f%%",
                                opp.direction.upper(), opp.symbol,
                                opp.funding_rate, opp.daily_income_pct)
            except Exception:
                logger.debug("Funding scan failed", exc_info=True)

            await asyncio.sleep(3600)  # every hour

    async def _run_polymarket_scanner(self) -> None:
        """Scan Polymarket every 30 minutes for mispriced markets."""
        from market_scanner.polymarket_scanner import scan_polymarket

        while self._running:
            try:
                opps = await scan_polymarket(min_delta=0.08, min_liquidity=1000, max_markets=10)
                for opp in opps[:3]:
                    signal_data = {
                        "symbol": f"POLY_{opp.condition_id[:12]}",
                        "direction": opp.direction,
                        "current_price": opp.market_price_yes,
                        "ml_win_probability": opp.confidence,
                        "smc_confidence": 0.0,
                        "setup_type": "polymarket_mispricing",
                        "signals": [f"delta={opp.delta:+.1%}", opp.reasoning[:50]],
                        "stop_loss": None,
                        "take_profit": None,
                        "asset_type": "prediction_market",
                        "question": opp.question,
                    }
                    await self._pipeline.process_signal(signal_data)
                    await self._journal.log_signal(signal_data)
                    await self._kg.add_signal(signal_data)
                    logger.info("Polymarket signal: %s  delta=%+.1f%%  \"%s\"",
                                opp.direction, opp.delta * 100, opp.question[:50])
            except Exception:
                logger.debug("Polymarket scan failed", exc_info=True)

            await asyncio.sleep(1800)  # every 30 min

    async def _run_cascade_monitor(self) -> None:
        """Monitor liquidation cascade risk every 5 minutes."""
        from market_scanner.liquidation_predictor import CascadePredictor

        predictor = CascadePredictor(self._redis)
        while self._running:
            try:
                risks = await predictor.scan()
                high_risk = [r for r in risks if r.risk_score >= 0.5]
                for r in high_risk:
                    direction = "short" if r.crowded_direction == "long" else "long"
                    signal_data = {
                        "symbol": r.symbol,
                        "direction": direction,
                        "current_price": 0,
                        "ml_win_probability": r.risk_score,
                        "smc_confidence": 0.0,
                        "setup_type": "cascade_risk",
                        "signals": r.signals[:5],
                        "stop_loss": None,
                        "take_profit": None,
                        "asset_type": "crypto_cascade",
                    }
                    await self._pipeline.process_signal(signal_data)
                    await self._journal.log_signal(signal_data)
                    await self._kg.add_signal(signal_data)
                    logger.info("Cascade signal: %s %s  risk=%.0f%%  crowd=%s %.0f%%",
                                direction.upper(), r.symbol,
                                r.risk_score * 100, r.crowded_direction, r.crowd_pct)
            except Exception:
                logger.debug("Cascade monitor failed", exc_info=True)

            await asyncio.sleep(300)  # every 5 min

    async def _run_kg_market_snapshots(self) -> None:
        """Embed a market state snapshot into the knowledge graph every hour."""
        while self._running:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10) as client:
                    fg_resp = await client.get("https://api.alternative.me/fng/?limit=1&format=json")
                    fg = int(fg_resp.json()["data"][0]["value"])

                    btc_resp = await client.get(
                        "https://api.binance.com/api/v3/ticker/24hr",
                        params={"symbol": "BTCUSDT"},
                    )
                    btc = btc_resp.json()
                    btc_price = float(btc["lastPrice"])
                    btc_change = float(btc["priceChangePercent"])

                    funding_resp = await client.get("https://fapi.binance.com/fapi/v1/premiumIndex")
                    rates = [(r["symbol"], abs(float(r.get("lastFundingRate", 0))), float(r.get("lastFundingRate", 0)))
                             for r in funding_resp.json() if r.get("lastFundingRate")]
                    rates.sort(key=lambda x: x[1], reverse=True)
                    top_sym, _, top_rate = rates[0] if rates else ("", 0, 0)

                snapshot = {
                    "fear_greed": fg,
                    "btc_price": btc_price,
                    "btc_24h_change": btc_change,
                    "top_funding_symbol": top_sym,
                    "top_funding_rate": top_rate * 100,
                    "regime": "risk_on" if fg > 55 else "risk_off" if fg < 45 else "neutral",
                }
                await self._kg.add_market_snapshot(snapshot)
                logger.info(
                    "KG snapshot: fg=%d  btc=$%.0f (%+.1f%%)  top_funding=%s %.3f%%",
                    fg, btc_price, btc_change, top_sym, top_rate * 100,
                )
            except Exception:
                logger.debug("KG market snapshot failed", exc_info=True)

            await asyncio.sleep(3600)  # every hour

    async def stop(self) -> None:
        self._running = False


async def main():
    import redis.asyncio as aioredis
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = aioredis.from_url(redis_url, decode_responses=True)

    try:
        from oracle_shared.db import init_db
        await init_db()
        logger.info("DB schema verified (CREATE IF NOT EXISTS)")
    except Exception:
        logger.warning("DB init failed — continuing without Postgres", exc_info=True)

    logger.info("=" * 55)
    logger.info("  ORACLE DAEMON — Always-On Trading Pipeline")
    logger.info("  Scanner + Journal + Paper Trading + Auto-Status")
    logger.info("=" * 55)

    daemon = OracleDaemon(redis_client)

    try:
        await daemon.start()
    except KeyboardInterrupt:
        await daemon.stop()
    finally:
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
