"""Expanded ML training — collects massive dataset from Binance + Yahoo, trains classifier."""
from __future__ import annotations

import asyncio
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs", "market-scanner"))

from market_scanner.providers.binance_ws import BinanceKlineProvider
from market_scanner.providers.stocks import StockProvider
from market_scanner.backtester import backtest_df
from market_scanner.ml_classifier import extract_features, FEATURE_NAMES


EXTENDED_FEATURES = FEATURE_NAMES + ["bias_is_bullish"]

CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "UNIUSDT", "AAVEUSDT", "LTCUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "SHIBUSDT", "PEPEUSDT",
]

STOCK_TICKERS = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "SPY", "QQQ", "COIN",
    "MSTR", "PLTR", "AMZN", "GOOGL", "META", "SOFI", "SQ",
    "MARA", "RIOT", "ARM", "SMCI", "PYPL",
]


def collect_trades_from_df(df, symbol, interval, max_hold):
    """Run backtest and extract features from OB retest trades."""
    result = backtest_df(df, symbol=symbol, interval=interval,
                         min_confidence=0.4, lookback_window=50, max_hold_bars=max_hold)
    rows = []
    for trade in result.trades:
        if trade.setup_type not in ("ob_retest", "fvg_fill"):
            continue
        feats = extract_features(df, trade.entry_bar, trade.direction)
        feats["bias_is_bullish"] = 1.0 if trade.direction == "long" else 0.0
        label = 1 if trade.pnl_pct > 0 else 0
        rows.append((feats, label, trade.pnl_pct))
    return rows


async def main():
    binance = BinanceKlineProvider()
    stocks = StockProvider()

    all_features = []
    all_labels = []
    all_pnls = []

    print("=" * 55)
    print("  EXPANDED ML TRAINING — FREE DATA ONLY")
    print("=" * 55)

    # Phase 1: Crypto 1h (1000 bars each)
    print("\nPhase 1: Crypto 1h (20 pairs x 1000 bars)")
    for sym in CRYPTO_PAIRS:
        try:
            candles = await binance.get_recent_klines(sym, interval="1h", limit=1000)
            if len(candles) < 100:
                continue
            df = pd.DataFrame([
                {"open": c.open, "high": c.high, "low": c.low,
                 "close": c.close, "volume": c.volume}
                for c in candles
            ])
            rows = collect_trades_from_df(df, sym, "1h", 48)
            for feats, label, pnl in rows:
                all_features.append(feats)
                all_labels.append(label)
                all_pnls.append(pnl)
            print(f"  {sym}: {len(rows)} trades from {len(candles)} bars")
        except Exception as e:
            print(f"  {sym}: failed - {str(e)[:40]}")

    c1 = len(all_features)

    # Phase 2: Crypto 4h (500 bars each)
    print(f"\nPhase 2: Crypto 4h (20 pairs x 500 bars)")
    for sym in CRYPTO_PAIRS:
        try:
            candles = await binance.get_recent_klines(sym, interval="4h", limit=500)
            if len(candles) < 100:
                continue
            df = pd.DataFrame([
                {"open": c.open, "high": c.high, "low": c.low,
                 "close": c.close, "volume": c.volume}
                for c in candles
            ])
            rows = collect_trades_from_df(df, sym, "4h", 24)
            for feats, label, pnl in rows:
                all_features.append(feats)
                all_labels.append(label)
                all_pnls.append(pnl)
            print(f"  {sym}: {len(rows)} trades")
        except Exception:
            pass

    c2 = len(all_features) - c1

    # Phase 3: Stocks daily (1 year)
    print(f"\nPhase 3: Stocks daily (20 tickers x 1 year)")
    for ticker in STOCK_TICKERS:
        try:
            df = await stocks.get_ohlcv(ticker, period="1y")
            if len(df) < 100:
                continue
            rows = collect_trades_from_df(df, ticker, "1d", 20)
            for feats, label, pnl in rows:
                all_features.append(feats)
                all_labels.append(label)
                all_pnls.append(pnl)
            print(f"  {ticker}: {len(rows)} trades from {len(df)} bars")
        except Exception:
            pass

    c3 = len(all_features) - c1 - c2
    total = len(all_features)
    wins = sum(all_labels)

    print(f"\n{'='*55}")
    print(f"TRAINING DATA")
    print(f"  Crypto 1h: {c1}  Crypto 4h: {c2}  Stocks daily: {c3}")
    print(f"  Total: {total} trades  Wins: {wins}  Losses: {total - wins}")
    print(f"  Raw win rate: {wins/total:.1%}")
    print(f"  Avg PnL: {np.mean(all_pnls):+.2f}%")
    print(f"{'='*55}")

    # Build arrays
    X = np.array([
        [row.get(name, 0.0) for name in EXTENDED_FEATURES]
        for row in all_features
    ])
    y = np.array(all_labels)
    pnls = np.array(all_pnls)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print("\nTraining gradient boosted classifier (300 trees, depth 5)...")
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    prec = cross_val_score(model, X, y, cv=cv, scoring="precision")
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")

    print(f"\nCROSS-VALIDATION (5-fold stratified)")
    print(f"  Accuracy:  {np.mean(acc):.1%} (+/- {np.std(acc):.1%})")
    print(f"  Precision: {np.mean(prec):.1%} (+/- {np.std(prec):.1%})")
    print(f"  F1 Score:  {np.mean(f1):.1%} (+/- {np.std(f1):.1%})")

    # Final train on all data
    model.fit(X, y)

    # Feature importance
    importances = sorted(
        zip(EXTENDED_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\nFEATURE IMPORTANCE:")
    for feat, imp in importances:
        bar = "#" * int(imp * 150)
        print(f"  {feat:<32} {imp:.4f} {bar}")

    # Save
    model_dir = Path(__file__).parent / ".." / "programs" / "market-scanner" / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "expanded_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": EXTENDED_FEATURES}, f)
    print(f"\nModel saved to {model_path}")

    # Simulated filtered trading
    print(f"\n{'='*55}")
    print("FILTERED TRADING SIMULATION")
    print("(only take trades where model predicts > threshold)")
    probas = model.predict_proba(X)[:, 1]

    print(f"\n  {'Threshold':<12} {'Trades':>7} {'Win Rate':>10} {'Avg PnL':>10} {'Total PnL':>12}")
    print(f"  {'-'*52}")
    print(f"  {'No filter':<12} {len(y):>7} {y.mean():>10.1%} {pnls.mean():>10.2f}% {pnls.sum():>11.1f}%")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = probas >= thresh
        if mask.sum() < 5:
            continue
        wr = y[mask].mean()
        avg = pnls[mask].mean()
        tot = pnls[mask].sum()
        print(f"  >={thresh:<10.0%} {mask.sum():>7} {wr:>10.1%} {avg:>10.2f}% {tot:>11.1f}%")

    print(f"{'='*55}")


if __name__ == "__main__":
    asyncio.run(main())
