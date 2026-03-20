"""One-command market snapshot — shows everything the system knows RIGHT NOW.

Run: python scripts/market_snapshot.py

Shows:
  - Market regime (crash/volatile/ranging/trending)
  - Fear & Greed index
  - Top convergence alerts
  - BTC correlation lag rankings
  - Crowded trades at cascade risk
  - Funding rate opportunities
  - News sentiment
  - Reddit trending
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs", "market-scanner"))

from market_scanner.providers.binance_ws import BinanceKlineProvider
from market_scanner.providers.scrapers import scrape_fear_greed, scrape_funding_rates, scrape_reddit_trending
from market_scanner.regime_detector import detect_regime
from market_scanner.correlation_lag import LagDetector, LEADER, FOLLOWERS
from market_scanner.liquidation_predictor import fetch_positioning, assess_cascade_risk
from market_scanner.funding_scanner import scan_funding
from market_scanner.news_sentiment import NewsSentiment
from market_scanner.smc import analyze_smc
from market_scanner.multi_timeframe import MultiTimeframe
import pandas as pd


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]


async def main():
    t0 = time.time()
    binance = BinanceKlineProvider()

    print()
    print("  $$$$$$\\  $$$$$$$\\   $$$$$$\\   $$$$$$\\  $$\\       $$$$$$$$\\ ")
    print("  $$  __$$\\ $$  __$$\\ $$  __$$\\ $$  __$$\\ $$ |      $$  _____|")
    print("  $$ /  $$ |$$ |  $$ |$$ /  $$ |$$ /  \\__|$$ |      $$ |      ")
    print("  $$ |  $$ |$$$$$$$  |$$$$$$$$ |$$ |      $$ |      $$$$$\\    ")
    print("  $$ |  $$ |$$  __$$< $$  __$$ |$$ |      $$ |      $$  __|   ")
    print("  $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |  $$\\ $$ |      $$ |     ")
    print("  $$$$$$  |$$ |  $$ |$$ |  $$ |\\$$$$$$  |$$$$$$$$\\ $$$$$$$$\\ ")
    print("  \\______/ \\__|  \\__|\\__|  \\__| \\______/ \\________|\\________|")
    print()
    print("=" * 65)
    print("  MARKET SNAPSHOT — Everything We Know Right Now")
    print("=" * 65)

    # 1. Fear & Greed
    print("\n1. MARKET SENTIMENT")
    fg = await scrape_fear_greed()
    if fg:
        bar_len = fg.value // 5
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"   Fear & Greed: {fg.value}/100 ({fg.label})")
        print(f"   [{bar}]")
        print(f"   Yesterday: {fg.yesterday}  Last week: {fg.last_week}  Last month: {fg.last_month}")
    sys.stdout.flush()

    # 2. Market Regimes
    print("\n2. MARKET REGIMES")
    for sym in SYMBOLS[:5]:
        try:
            candles = await binance.get_recent_klines(sym, interval="1d", limit=250)
            df = pd.DataFrame([{"open":c.open,"high":c.high,"low":c.low,"close":c.close,"volume":c.volume} for c in candles])
            r = detect_regime(df)
            icon = {"trending_up": "^", "trending_down": "v", "ranging": "-", "high_volatility": "!", "crash": "X", "recovery": "~"}.get(r.regime.value, "?")
            print(f"   {sym:<12} [{icon}] {r.regime.value:<16} size={r.position_size_multiplier:.1f}x  strategy={r.preferred_strategy}")
        except Exception:
            pass
    sys.stdout.flush()

    # 3. SMC + MTF Quick Scan
    print("\n3. TRADE SETUPS (SMC + Multi-Timeframe)")
    mtf = MultiTimeframe()
    for sym in SYMBOLS:
        try:
            result = await mtf.analyze_crypto(sym, binance, [("4h", 100), ("1h", 100), ("15m", 100)])
            if result.aligned:
                candles = await binance.get_recent_klines(sym, interval="15m", limit=100)
                df = pd.DataFrame([{"open":c.open,"high":c.high,"low":c.low,"close":c.close,"volume":c.volume} for c in candles])
                smc = analyze_smc(df, sym, "crypto")
                setup = smc.setup_type if smc else "none"
                conf = smc.confidence if smc else 0
                print(f"   {sym:<12} {result.direction.upper():<6} align={result.alignment_score:.0%} smc={conf:.0%} setup={setup}")
        except Exception:
            pass
    sys.stdout.flush()

    # 4. BTC Correlation
    print("\n4. BTC CORRELATION LAG (if BTC moves, who follows?)")
    detector = LagDetector(interval="1m")
    await detector._backfill([LEADER] + FOLLOWERS[:8])
    detector._recompute_profiles()
    rankings = detector.get_rankings()[:5]
    for sym, corr, lag, beta in rankings:
        print(f"   {sym:<12} corr={corr:+.2f}  lag={lag}min  beta={beta:+.2f}x")
    sys.stdout.flush()

    # 5. Cascade Risk
    print("\n5. LIQUIDATION CASCADE RISK")
    positions = await fetch_positioning(SYMBOLS)
    risks = assess_cascade_risk(positions)
    for r in risks[:5]:
        if r.risk_score > 0:
            crowd = r.crowded_direction.upper()[:5] if r.crowded_direction != "neutral" else "---"
            print(f"   {r.symbol:<12} risk={r.risk_score:.2f}  crowd={crowd} {r.crowd_pct:.0f}%  cascade={r.expected_cascade_pct:.1f}%")
    sys.stdout.flush()

    # 6. Funding Opportunities
    print("\n6. TOP FUNDING RATE PLAYS")
    funding = await scan_funding(min_rate_pct=0.05, min_volume_usd=5_000_000)
    for f in funding[:5]:
        print(f"   {f.symbol:<14} {f.direction:<6} rate={f.funding_rate:+.3f}%/8h  daily={f.daily_income_pct:.2f}%  annual={f.annualized_pct:.0f}%")
    sys.stdout.flush()

    # 7. News
    print("\n7. NEWS SENTIMENT")
    ns = NewsSentiment()
    for ticker in ["BTC", "TSLA", "NVDA"]:
        try:
            result = await ns.analyze(ticker, use_gemini=False, limit=5)
            icon = "+" if result.score > 0.1 else "-" if result.score < -0.1 else "~"
            print(f"   {ticker:<8} [{icon}] score={result.score:+.2f}  ({result.bullish_count}B/{result.bearish_count}b/{result.neutral_count}N)")
        except Exception:
            pass
    sys.stdout.flush()

    # 8. Reddit
    print("\n8. REDDIT TRENDING")
    trends = await scrape_reddit_trending(["wallstreetbets", "cryptocurrency"])
    for t in trends[:5]:
        tickers = f" [{','.join(t.tickers_mentioned)}]" if t.tickers_mentioned else ""
        print(f"   r/{t.subreddit}: {t.title[:55]}...{tickers}")
    sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Snapshot complete in {elapsed:.0f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    asyncio.run(main())
