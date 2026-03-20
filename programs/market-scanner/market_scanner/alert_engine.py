"""Alert engine — aggregates all signals and emits high-conviction alerts.

Combines every signal source into one unified alert:
  - SMC pattern + ML model prediction
  - Multi-timeframe alignment
  - News sentiment
  - Correlation lag (BTC moved, alt about to follow)
  - Liquidation cascade risk
  - Order book imbalance
  - Funding rate extremes

When multiple signals converge on the same trade, that's a high-conviction
setup. This engine scores the convergence and emits alerts via:
  - Redis pub/sub (for dashboard)
  - Telegram (for mobile)
  - Console logging (for monitoring)

Usage::

    engine = AlertEngine(redis)
    await engine.run_scan()  # one-shot scan
    await engine.monitor()   # continuous loop
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

ALERT_CHANNEL = "oracle:trade_alert"


@dataclass
class ConvergenceAlert:
    """A high-conviction trade alert where multiple signals converge."""
    symbol: str
    direction: str              # "long" | "short"
    conviction: float           # 0-1, composite score
    entry_price: float
    stop_loss: float | None
    take_profit: float | None

    # Individual signal scores
    smc_score: float            # SMC pattern confidence
    ml_probability: float       # ML model win probability
    mtf_alignment: float        # multi-timeframe alignment (0-1)
    news_sentiment: float       # -1 to +1
    lag_signal: bool            # BTC just moved, this alt follows
    cascade_risk: float         # 0-1, risk of liquidation cascade (in our favor)
    ob_imbalance: float         # order book imbalance (-1 to +1)

    # Context
    signals: list[str] = field(default_factory=list)
    timeframe: str = ""
    timestamp: str = ""

    # Regime-adjusted sizing
    regime: str = ""                 # "ranging", "high_volatility", etc.
    position_size_mult: float = 1.0  # regime-adjusted position multiplier
    stop_loss_mult: float = 1.0     # regime-adjusted stop width multiplier
    suggested_position_usd: float = 0.0  # for $10k portfolio

    def format_alert(self) -> str:
        """Format as a readable alert message."""
        d = "LONG" if self.direction == "long" else "SHORT"
        lines = [
            f"{'='*40}",
            f"  {d} {self.symbol}  conviction={self.conviction:.0%}",
            f"{'='*40}",
            f"  Entry: ${self.entry_price:.4f}",
        ]
        if self.stop_loss:
            lines.append(f"  Stop:  ${self.stop_loss:.4f}")
        if self.take_profit:
            lines.append(f"  TP:    ${self.take_profit:.4f}")
        lines.append("")
        lines.append("  Signal convergence:")
        if self.smc_score > 0:
            lines.append(f"    SMC pattern:    {self.smc_score:.0%}")
        if self.ml_probability > 0.5:
            lines.append(f"    ML prediction:  {self.ml_probability:.0%} win prob")
        if self.mtf_alignment > 0:
            lines.append(f"    Multi-TF:       {self.mtf_alignment:.0%} aligned")
        if abs(self.news_sentiment) > 0.1:
            sent = "bullish" if self.news_sentiment > 0 else "bearish"
            lines.append(f"    News sentiment: {sent} ({self.news_sentiment:+.2f})")
        if self.lag_signal:
            lines.append(f"    BTC lag signal: YES (alt following BTC move)")
        if self.cascade_risk > 0.3:
            lines.append(f"    Cascade risk:   {self.cascade_risk:.0%} (liquidations favor us)")
        if abs(self.ob_imbalance) > 0.2:
            side = "buy" if self.ob_imbalance > 0 else "sell"
            lines.append(f"    Order book:     {side} pressure ({self.ob_imbalance:+.2f})")
        lines.append(f"\n  Signals: {', '.join(self.signals[:5])}")
        if self.regime:
            lines.append(f"\n  Regime: {self.regime.upper()}")
            lines.append(f"  Position: {self.position_size_mult:.1f}x normal (${self.suggested_position_usd:,.0f} on $10k)")
            if self.stop_loss_mult != 1.0:
                lines.append(f"  Stop width: {self.stop_loss_mult:.1f}x normal")
        return "\n".join(lines)


class AlertEngine:
    """Aggregate all signal sources and emit high-conviction alerts."""

    def __init__(
        self,
        redis_client: Any | None = None,
        min_conviction: float = 0.5,
    ) -> None:
        self._redis = redis_client
        self._min_conviction = min_conviction

    async def run_scan(
        self,
        symbols: list[str] | None = None,
    ) -> list[ConvergenceAlert]:
        """Run a full scan cycle across all signal sources."""
        symbols = symbols or [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "APTUSDT", "NEARUSDT",
        ]

        # Collect all signal sources in parallel
        smc_data, mtf_data, news_data, cascade_data, ob_data, ls_data, regime_data = (
            await asyncio.gather(
                self._get_smc_signals(symbols),
                self._get_mtf_signals(symbols),
                self._get_news_signals(symbols),
                self._get_cascade_signals(symbols),
                self._get_orderbook_signals(symbols),
                self._get_ls_ratios(symbols),
                self._get_regime_data(symbols),
                return_exceptions=True,
            )
        )

        # Handle exceptions
        if isinstance(smc_data, Exception):
            smc_data = {}
        if isinstance(mtf_data, Exception):
            mtf_data = {}
        if isinstance(news_data, Exception):
            news_data = {}
        if isinstance(cascade_data, Exception):
            cascade_data = {}
        if isinstance(ob_data, Exception):
            ob_data = {}
        if isinstance(ls_data, Exception):
            ls_data = {}
        if isinstance(regime_data, Exception):
            regime_data = {}

        # Compute convergence for each symbol
        alerts: list[ConvergenceAlert] = []
        for sym in symbols:
            alert = self._compute_convergence(
                sym, smc_data, mtf_data, news_data, cascade_data, ob_data, ls_data,
            )
            if alert and alert.conviction >= self._min_conviction:
                # Apply regime adjustments
                regime = regime_data.get(sym, {})
                if regime:
                    alert.regime = regime.get("regime", "")
                    alert.position_size_mult = regime.get("pos_mult", 1.0)
                    alert.stop_loss_mult = regime.get("sl_mult", 1.0)
                    base_position = 10000 * 0.05  # 5% of $10k
                    alert.suggested_position_usd = round(base_position * alert.position_size_mult, 0)
            if alert and alert.conviction >= self._min_conviction:
                alerts.append(alert)
                if self._redis:
                    await self._publish(alert)

        alerts.sort(key=lambda a: a.conviction, reverse=True)
        return alerts

    def _compute_convergence(
        self,
        symbol: str,
        smc_data: dict,
        mtf_data: dict,
        news_data: dict,
        cascade_data: dict,
        ob_data: dict,
        ls_data: dict,
    ) -> ConvergenceAlert | None:
        """Score signal convergence for a single symbol."""
        smc = smc_data.get(symbol, {})
        mtf = mtf_data.get(symbol, {})
        news = news_data.get(symbol, {})
        cascade = cascade_data.get(symbol, {})
        ob = ob_data.get(symbol, {})
        ls = ls_data.get(symbol, {})

        if not smc and not mtf:
            return None

        # Determine direction from strongest signals
        bull_score = 0.0
        bear_score = 0.0
        signals: list[str] = []

        # SMC bias
        smc_conf = smc.get("confidence", 0)
        smc_bias = smc.get("bias", "neutral")
        if smc_bias == "bullish":
            bull_score += smc_conf * 0.25
            signals.append(f"smc_bullish_{smc_conf:.0%}")
        elif smc_bias == "bearish":
            bear_score += smc_conf * 0.25
            signals.append(f"smc_bearish_{smc_conf:.0%}")

        # MTF alignment
        mtf_dir = mtf.get("direction", "none")
        mtf_align = mtf.get("alignment", 0)
        if mtf_dir == "long":
            bull_score += mtf_align * 0.25
            if mtf_align >= 0.75:
                signals.append(f"mtf_aligned_bull_{mtf_align:.0%}")
        elif mtf_dir == "short":
            bear_score += mtf_align * 0.25
            if mtf_align >= 0.75:
                signals.append(f"mtf_aligned_bear_{mtf_align:.0%}")

        # News sentiment
        news_score = news.get("score", 0)
        if news_score > 0.15:
            bull_score += min(0.15, news_score * 0.3)
            signals.append("news_bullish")
        elif news_score < -0.15:
            bear_score += min(0.15, abs(news_score) * 0.3)
            signals.append("news_bearish")

        # Cascade risk (favorable if crowd is wrong)
        cascade_risk = cascade.get("risk", 0)
        crowd_dir = cascade.get("crowded", "neutral")
        if cascade_risk > 0.3:
            if crowd_dir == "long":
                bear_score += cascade_risk * 0.15
                signals.append("cascade_risk_longs")
            elif crowd_dir == "short":
                bull_score += cascade_risk * 0.15
                signals.append("cascade_risk_shorts")

        # Order book imbalance
        imbalance = ob.get("imbalance", 0)
        if imbalance > 0.2:
            bull_score += 0.1
            signals.append("ob_buy_pressure")
        elif imbalance < -0.2:
            bear_score += 0.1
            signals.append("ob_sell_pressure")

        # Determine direction and conviction
        if bull_score > bear_score and bull_score > 0.2:
            direction = "long"
            conviction = min(1.0, bull_score)
        elif bear_score > bull_score and bear_score > 0.2:
            direction = "short"
            conviction = min(1.0, bear_score)
        else:
            return None

        return ConvergenceAlert(
            symbol=symbol,
            direction=direction,
            conviction=round(conviction, 3),
            entry_price=smc.get("price", 0),
            stop_loss=smc.get("stop_loss"),
            take_profit=smc.get("take_profit"),
            smc_score=smc_conf,
            ml_probability=smc.get("ml_prob", 0.5),
            mtf_alignment=mtf_align,
            news_sentiment=news_score,
            lag_signal=False,
            cascade_risk=cascade_risk,
            ob_imbalance=imbalance,
            signals=signals,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def _get_smc_signals(self, symbols: list[str]) -> dict:
        """Get SMC analysis for each symbol."""
        from market_scanner.providers.binance_ws import BinanceKlineProvider
        from market_scanner.smc import analyze_smc, Bias
        import pandas as pd

        provider = BinanceKlineProvider()
        results = {}
        for sym in symbols:
            try:
                candles = await provider.get_recent_klines(sym, interval="15m", limit=100)
                if len(candles) < 30:
                    continue
                df = pd.DataFrame([
                    {"open": c.open, "high": c.high, "low": c.low,
                     "close": c.close, "volume": c.volume}
                    for c in candles
                ])
                smc = analyze_smc(df, sym, "crypto")
                if smc:
                    results[sym] = {
                        "bias": smc.bias.value,
                        "confidence": smc.confidence,
                        "setup": smc.setup_type or "none",
                        "price": smc.current_price,
                        "stop_loss": smc.stop_loss,
                        "take_profit": smc.take_profit,
                    }
            except Exception:
                pass
        return results

    async def _get_mtf_signals(self, symbols: list[str]) -> dict:
        """Get multi-timeframe alignment."""
        from market_scanner.multi_timeframe import MultiTimeframe
        from market_scanner.providers.binance_ws import BinanceKlineProvider

        mtf = MultiTimeframe()
        provider = BinanceKlineProvider()
        results = {}
        for sym in symbols:
            try:
                result = await mtf.analyze_crypto(sym, provider, [("4h", 100), ("1h", 100), ("15m", 100)])
                results[sym] = {
                    "direction": result.direction,
                    "alignment": result.alignment_score,
                    "confidence": result.confidence,
                }
            except Exception:
                pass
        return results

    async def _get_news_signals(self, symbols: list[str]) -> dict:
        """Get news sentiment — only for top symbols to conserve Gemini quota."""
        from market_scanner.news_sentiment import NewsSentiment
        ns = NewsSentiment()
        results = {}
        # Map Binance symbols to ticker names for news search
        ticker_map = {
            "BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "SOLUSDT": "Solana",
            "BNBUSDT": "BNB", "XRPUSDT": "XRP", "DOGEUSDT": "Dogecoin",
            "AVAXUSDT": "Avalanche AVAX", "LINKUSDT": "Chainlink",
            "APTUSDT": "Aptos", "NEARUSDT": "NEAR Protocol",
        }
        for sym in symbols[:5]:  # limit to 5 to conserve API
            query = ticker_map.get(sym, sym.replace("USDT", ""))
            try:
                result = await ns.analyze(query, use_gemini=False, limit=5)
                results[sym] = {"score": result.score}
            except Exception:
                pass
        return results

    async def _get_cascade_signals(self, symbols: list[str]) -> dict:
        """Get liquidation cascade risk."""
        from market_scanner.liquidation_predictor import fetch_positioning, assess_cascade_risk
        try:
            positions = await fetch_positioning(symbols)
            risks = assess_cascade_risk(positions)
            return {
                r.symbol: {
                    "risk": r.risk_score,
                    "crowded": r.crowded_direction,
                    "cascade_pct": r.expected_cascade_pct,
                }
                for r in risks
            }
        except Exception:
            return {}

    async def _get_orderbook_signals(self, symbols: list[str]) -> dict:
        """Get order book imbalance."""
        import httpx
        results = {}
        async with httpx.AsyncClient(timeout=10) as client:
            for sym in symbols:
                try:
                    resp = await client.get(
                        f"https://api.binance.com/api/v3/depth",
                        params={"symbol": sym, "limit": 20},
                    )
                    data = resp.json()
                    bids = sum(float(q) for _, q in data.get("bids", []))
                    asks = sum(float(q) for _, q in data.get("asks", []))
                    total = bids + asks
                    results[sym] = {
                        "imbalance": round((bids - asks) / total, 3) if total > 0 else 0,
                    }
                    await asyncio.sleep(0.05)
                except Exception:
                    pass
        return results

    async def _get_ls_ratios(self, symbols: list[str]) -> dict:
        """Get long/short ratios."""
        import httpx
        results = {}
        async with httpx.AsyncClient(timeout=10) as client:
            for sym in symbols:
                try:
                    resp = await client.get(
                        "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
                        params={"symbol": sym, "period": "1h", "limit": 1},
                    )
                    data = resp.json()
                    if data and isinstance(data, list):
                        results[sym] = {
                            "ratio": float(data[0].get("longShortRatio", 1)),
                        }
                    await asyncio.sleep(0.05)
                except Exception:
                    pass
        return results

    async def _get_regime_data(self, symbols: list[str]) -> dict:
        """Get market regime for each symbol."""
        from market_scanner.providers.binance_ws import BinanceKlineProvider
        from market_scanner.regime_detector import detect_regime
        import pandas as pd

        provider = BinanceKlineProvider()
        results = {}
        for sym in symbols:
            try:
                candles = await provider.get_recent_klines(sym, interval="1d", limit=250)
                if len(candles) < 50:
                    continue
                df = pd.DataFrame([
                    {"open": c.open, "high": c.high, "low": c.low,
                     "close": c.close, "volume": c.volume}
                    for c in candles
                ])
                regime = detect_regime(df)
                results[sym] = {
                    "regime": regime.regime.value,
                    "pos_mult": regime.position_size_multiplier,
                    "sl_mult": regime.stop_loss_multiplier,
                    "strategy": regime.preferred_strategy,
                }
            except Exception:
                pass
        return results

    async def _publish(self, alert: ConvergenceAlert) -> None:
        """Publish alert to Redis."""
        if self._redis:
            await self._redis.publish(ALERT_CHANNEL, json.dumps({
                "symbol": alert.symbol,
                "direction": alert.direction,
                "conviction": alert.conviction,
                "entry_price": alert.entry_price,
                "stop_loss": alert.stop_loss,
                "take_profit": alert.take_profit,
                "signals": alert.signals,
                "timestamp": alert.timestamp,
            }))
