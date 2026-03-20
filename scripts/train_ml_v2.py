"""V2 ML Training — enhanced features + more data + multi-timeframe.

New features over v1:
  - Funding rate (from Binance futures)
  - Fear & Greed index
  - Multi-timeframe: 1h trend vs 4h trend alignment
  - Cross-asset correlation (BTC as market leader)
  - Time features (hour of day, day of week)
  - Relative volume (vs 5-bar and 50-bar averages)
  - Price position in range (0-1 where in the 20-bar high-low range)

More data:
  - 30 crypto pairs (up from 20)
  - 1h + 4h + 15m timeframes
  - 25 stock tickers (up from 20)
"""
from __future__ import annotations

import asyncio
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs", "market-scanner"))

from market_scanner.providers.binance_ws import BinanceKlineProvider
from market_scanner.providers.stocks import StockProvider
from market_scanner.providers.scrapers import scrape_fear_greed, scrape_funding_rates
from market_scanner.backtester import backtest_df
from market_scanner.ml_classifier import extract_features, FEATURE_NAMES
from market_scanner.smc import find_swing_points, detect_structure


# V2 features = V1 + new ones
V2_FEATURES = FEATURE_NAMES + [
    "bias_is_bullish",
    # New in v2:
    "funding_rate",           # current funding rate for this asset
    "fear_greed",             # market-wide fear/greed (0-100)
    "higher_tf_aligned",      # 1 if higher timeframe trend agrees with setup direction
    "btc_trend_bullish",      # 1 if BTC is in uptrend (market leader)
    "hour_sin",               # cyclical hour encoding (sin)
    "hour_cos",               # cyclical hour encoding (cos)
    "day_of_week",            # 0-6
    "vol_ratio_5",            # volume / 5-bar average
    "vol_ratio_50",           # volume / 50-bar average
    "range_position",         # where in 20-bar high-low range (0=low, 1=high)
    "body_upper_wick_ratio",  # upper wick / body (rejection strength)
    "body_lower_wick_ratio",  # lower wick / body
    "rsi_slope",              # RSI change over last 3 bars
    "macd_above_signal",      # 1 if MACD > signal line
]

CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "UNIUSDT", "AAVEUSDT", "LTCUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "SHIBUSDT", "PEPEUSDT",
    "SUIUSDT", "FILUSDT", "INJUSDT", "FETUSDT", "RENDERUSDT",
    "TIAUSDT", "JUPUSDT", "WUSDT", "PENDLEUSDT", "ENAUSDT",
]

STOCK_TICKERS = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "SPY", "QQQ", "COIN",
    "MSTR", "PLTR", "AMZN", "GOOGL", "META", "SOFI", "PYPL",
    "MARA", "RIOT", "ARM", "SMCI", "INTC", "NFLX", "CRM", "UBER",
    "SHOP", "RBLX",
]


def extract_v2_features(
    df: pd.DataFrame,
    bar_index: int,
    direction: str,
    funding_rate: float = 0.0,
    fear_greed: int = 50,
    higher_tf_bullish: bool | None = None,
    btc_bullish: bool | None = None,
) -> dict[str, float]:
    """Extract V2 feature set — V1 features + new market context features."""
    # Start with V1 features
    feats = extract_features(df, bar_index, direction)
    feats["bias_is_bullish"] = 1.0 if direction == "long" else 0.0

    # New features
    feats["funding_rate"] = funding_rate * 10000  # scale to basis points
    feats["fear_greed"] = float(fear_greed)
    feats["higher_tf_aligned"] = 1.0 if higher_tf_bullish is not None and (
        (higher_tf_bullish and direction == "long") or
        (not higher_tf_bullish and direction == "short")
    ) else 0.0
    feats["btc_trend_bullish"] = 1.0 if btc_bullish else 0.0

    # Time features (use bar index as proxy if no timestamps)
    feats["hour_sin"] = np.sin(2 * np.pi * (bar_index % 24) / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * (bar_index % 24) / 24)
    feats["day_of_week"] = float(bar_index % 7)

    if bar_index >= 50 and bar_index < len(df):
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Volume ratios at different scales
        vol_now = float(volume.iloc[bar_index])
        vol_5 = float(volume.iloc[max(0, bar_index-5):bar_index+1].mean())
        vol_50 = float(volume.iloc[max(0, bar_index-50):bar_index+1].mean())
        feats["vol_ratio_5"] = vol_now / vol_5 if vol_5 > 0 else 1.0
        feats["vol_ratio_50"] = vol_now / vol_50 if vol_50 > 0 else 1.0

        # Range position
        high_20 = float(high.iloc[max(0, bar_index-20):bar_index+1].max())
        low_20 = float(low.iloc[max(0, bar_index-20):bar_index+1].min())
        rng = high_20 - low_20
        feats["range_position"] = (float(close.iloc[bar_index]) - low_20) / rng if rng > 0 else 0.5

        # Wick ratios
        c = float(close.iloc[bar_index])
        o = float(open_.iloc[bar_index])
        h = float(high.iloc[bar_index])
        l = float(low.iloc[bar_index])
        body = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        feats["body_upper_wick_ratio"] = upper_wick / body if body > 0 else 0
        feats["body_lower_wick_ratio"] = lower_wick / body if body > 0 else 0

        # RSI slope
        rsi = ta.momentum.RSIIndicator(close.iloc[:bar_index+1], window=14).rsi()
        if len(rsi) >= 4:
            feats["rsi_slope"] = float(rsi.iloc[-1]) - float(rsi.iloc[-4])
        else:
            feats["rsi_slope"] = 0.0

        # MACD above signal
        macd_ind = ta.trend.MACD(close.iloc[:bar_index+1])
        macd_line = macd_ind.macd()
        signal_line = macd_ind.macd_signal()
        if len(macd_line) > 0 and len(signal_line) > 0:
            feats["macd_above_signal"] = 1.0 if float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]) else 0.0
        else:
            feats["macd_above_signal"] = 0.0
    else:
        for f in ["vol_ratio_5", "vol_ratio_50", "range_position",
                   "body_upper_wick_ratio", "body_lower_wick_ratio",
                   "rsi_slope", "macd_above_signal"]:
            feats[f] = 0.0

    return feats


async def main():
    binance = BinanceKlineProvider()
    stocks = StockProvider()

    all_features = []
    all_labels = []
    all_pnls = []

    print("=" * 60)
    print("  V2 ML TRAINING — Enhanced Features + More Data")
    print("=" * 60)

    # Pre-fetch market context
    print("\nFetching market context...")
    fg = await scrape_fear_greed()
    fear_greed = fg.value if fg else 50
    print(f"  Fear & Greed: {fear_greed}")

    funding = await scrape_funding_rates()
    funding_map = {r.symbol: r.rate for r in funding}
    print(f"  Funding rates: {len(funding_map)} pairs")

    # Get BTC trend for cross-asset correlation
    btc_candles = await binance.get_recent_klines("BTCUSDT", interval="4h", limit=100)
    btc_df = pd.DataFrame([{"open":c.open,"high":c.high,"low":c.low,"close":c.close,"volume":c.volume} for c in btc_candles])
    btc_highs, btc_lows = find_swing_points(btc_df["high"], btc_df["low"], lookback=5)
    btc_trend, _ = detect_structure(btc_highs, btc_lows, btc_df["close"])
    btc_bullish = btc_trend == "uptrend"
    print(f"  BTC 4h trend: {btc_trend}")

    # ── Crypto 1h ─────────────────────────────────────────────────────────
    print(f"\nPhase 1: Crypto 1h (30 pairs x 1000 bars)")
    for sym in CRYPTO_PAIRS:
        try:
            candles = await binance.get_recent_klines(sym, interval="1h", limit=1000)
            if len(candles) < 100:
                continue
            df = pd.DataFrame([{"open":c.open,"high":c.high,"low":c.low,"close":c.close,"volume":c.volume} for c in candles])

            # Get 4h trend for this asset (higher TF alignment)
            candles_4h = await binance.get_recent_klines(sym, interval="4h", limit=100)
            df_4h = pd.DataFrame([{"open":c.open,"high":c.high,"low":c.low,"close":c.close,"volume":c.volume} for c in candles_4h])
            sh4, sl4 = find_swing_points(df_4h["high"], df_4h["low"], lookback=5)
            trend_4h, _ = detect_structure(sh4, sl4, df_4h["close"])
            htf_bull = trend_4h == "uptrend"

            fr = funding_map.get(sym, 0.0)

            result = backtest_df(df, symbol=sym, interval="1h", min_confidence=0.4, lookback_window=50, max_hold_bars=48)
            for trade in result.trades:
                if trade.setup_type not in ("ob_retest", "fvg_fill"):
                    continue
                feats = extract_v2_features(
                    df, trade.entry_bar, trade.direction,
                    funding_rate=fr, fear_greed=fear_greed,
                    higher_tf_bullish=htf_bull, btc_bullish=btc_bullish,
                )
                all_features.append(feats)
                all_labels.append(1 if trade.pnl_pct > 0 else 0)
                all_pnls.append(trade.pnl_pct)
            print(f"  {sym}: {len([t for t in result.trades if t.setup_type in ('ob_retest','fvg_fill')])} trades  4h={trend_4h}  fr={fr*100:+.3f}%")
        except Exception as e:
            print(f"  {sym}: {str(e)[:40]}")

    c1 = len(all_features)

    # ── Crypto 4h ─────────────────────────────────────────────────────────
    print(f"\nPhase 2: Crypto 4h (30 pairs x 500 bars)")
    for sym in CRYPTO_PAIRS:
        try:
            candles = await binance.get_recent_klines(sym, interval="4h", limit=500)
            if len(candles) < 100:
                continue
            df = pd.DataFrame([{"open":c.open,"high":c.high,"low":c.low,"close":c.close,"volume":c.volume} for c in candles])
            fr = funding_map.get(sym, 0.0)

            result = backtest_df(df, symbol=sym, interval="4h", min_confidence=0.4, lookback_window=50, max_hold_bars=24)
            for trade in result.trades:
                if trade.setup_type not in ("ob_retest", "fvg_fill"):
                    continue
                feats = extract_v2_features(
                    df, trade.entry_bar, trade.direction,
                    funding_rate=fr, fear_greed=fear_greed,
                    btc_bullish=btc_bullish,
                )
                all_features.append(feats)
                all_labels.append(1 if trade.pnl_pct > 0 else 0)
                all_pnls.append(trade.pnl_pct)
            count = len([t for t in result.trades if t.setup_type in ("ob_retest", "fvg_fill")])
            print(f"  {sym}: {count} trades")
        except Exception:
            pass

    c2 = len(all_features) - c1

    # ── Crypto 15m (scalping timeframe) ───────────────────────────────────
    print(f"\nPhase 3: Crypto 15m (10 major pairs x 1000 bars)")
    for sym in CRYPTO_PAIRS[:10]:
        try:
            candles = await binance.get_recent_klines(sym, interval="15m", limit=1000)
            if len(candles) < 100:
                continue
            df = pd.DataFrame([{"open":c.open,"high":c.high,"low":c.low,"close":c.close,"volume":c.volume} for c in candles])
            fr = funding_map.get(sym, 0.0)

            result = backtest_df(df, symbol=sym, interval="15m", min_confidence=0.4, lookback_window=50, max_hold_bars=96)
            for trade in result.trades:
                if trade.setup_type not in ("ob_retest", "fvg_fill"):
                    continue
                feats = extract_v2_features(
                    df, trade.entry_bar, trade.direction,
                    funding_rate=fr, fear_greed=fear_greed,
                    btc_bullish=btc_bullish,
                )
                all_features.append(feats)
                all_labels.append(1 if trade.pnl_pct > 0 else 0)
                all_pnls.append(trade.pnl_pct)
            count = len([t for t in result.trades if t.setup_type in ("ob_retest", "fvg_fill")])
            print(f"  {sym}: {count} trades")
        except Exception:
            pass

    c3 = len(all_features) - c1 - c2

    # ── Stocks daily ──────────────────────────────────────────────────────
    print(f"\nPhase 4: Stocks daily (25 tickers x 1 year)")
    for ticker in STOCK_TICKERS:
        try:
            df = await stocks.get_ohlcv(ticker, period="1y")
            if len(df) < 100:
                continue
            result = backtest_df(df, symbol=ticker, interval="1d", min_confidence=0.4, lookback_window=50, max_hold_bars=20)
            for trade in result.trades:
                if trade.setup_type not in ("ob_retest", "fvg_fill"):
                    continue
                feats = extract_v2_features(
                    df, trade.entry_bar, trade.direction,
                    fear_greed=fear_greed, btc_bullish=btc_bullish,
                )
                all_features.append(feats)
                all_labels.append(1 if trade.pnl_pct > 0 else 0)
                all_pnls.append(trade.pnl_pct)
            count = len([t for t in result.trades if t.setup_type in ("ob_retest", "fvg_fill")])
            print(f"  {ticker}: {count} trades from {len(df)} bars")
        except Exception:
            pass

    c4 = len(all_features) - c1 - c2 - c3
    total = len(all_features)
    wins = sum(all_labels)

    print(f"\n{'='*60}")
    print(f"V2 TRAINING DATA")
    print(f"  Crypto 1h: {c1}  4h: {c2}  15m: {c3}  Stocks: {c4}")
    print(f"  Total: {total} trades  Wins: {wins}  Losses: {total - wins}")
    print(f"  Raw win rate: {wins/total:.1%}")
    print(f"  Features per trade: {len(V2_FEATURES)}")
    print(f"{'='*60}")

    # Build arrays
    X = np.array([[row.get(name, 0.0) for name in V2_FEATURES] for row in all_features])
    y = np.array(all_labels)
    pnls = np.array(all_pnls)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print("\nTraining V2 model (500 trees, depth 6, 34 features)...")
    model = GradientBoostingClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=15, random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    prec = cross_val_score(model, X, y, cv=cv, scoring="precision")
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")

    print(f"\nV2 CROSS-VALIDATION (5-fold)")
    print(f"  Accuracy:  {np.mean(acc):.1%} (+/- {np.std(acc):.1%})")
    print(f"  Precision: {np.mean(prec):.1%} (+/- {np.std(prec):.1%})")
    print(f"  F1 Score:  {np.mean(f1):.1%} (+/- {np.std(f1):.1%})")

    model.fit(X, y)

    # Feature importance
    importances = sorted(zip(V2_FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\nV2 FEATURE IMPORTANCE:")
    for feat, imp in importances[:15]:
        bar = "#" * int(imp * 200)
        print(f"  {feat:<32} {imp:.4f} {bar}")

    # Save
    model_dir = Path(__file__).parent / ".." / "programs" / "market-scanner" / "models"
    model_dir.mkdir(exist_ok=True)
    with open(model_dir / "v2_expanded.pkl", "wb") as f:
        pickle.dump({"model": model, "features": V2_FEATURES, "version": 2}, f)
    print(f"\nV2 model saved")

    # Filtered trading sim
    print(f"\n{'='*60}")
    print("V2 FILTERED TRADING SIMULATION")
    probas = model.predict_proba(X)[:, 1]

    print(f"\n  {'Filter':<12} {'Trades':>7} {'Win Rate':>10} {'Avg PnL':>10} {'Total PnL':>12}")
    print(f"  {'-'*55}")
    print(f"  {'No filter':<12} {len(y):>7} {y.mean():>10.1%} {pnls.mean():>10.2f}% {pnls.sum():>11.1f}%")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = probas >= thresh
        if mask.sum() < 5:
            continue
        print(f"  >={thresh:<10.0%} {mask.sum():>7} {y[mask].mean():>10.1%} {pnls[mask].mean():>10.2f}% {pnls[mask].sum():>11.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
