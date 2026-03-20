"""Phase 1.1 FAST — Optimized historical pipeline.

Instead of walk-forward backtesting on 17k bars (slow), we:
1. Pull the full history
2. Split into overlapping 200-bar chunks
3. Run backtest on each chunk independently
4. Much faster: O(chunks * 200) vs O(17000 * 200)
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import pickle
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs", "market-scanner"))

from market_scanner.backtester import backtest_df
from market_scanner.ml_classifier import extract_features, FEATURE_NAMES

V2_FEATURES = FEATURE_NAMES + [
    "bias_is_bullish", "funding_rate", "fear_greed", "higher_tf_aligned",
    "btc_trend_bullish", "hour_sin", "hour_cos", "day_of_week",
    "vol_ratio_5", "vol_ratio_50", "range_position",
    "body_upper_wick_ratio", "body_lower_wick_ratio",
    "rsi_slope", "macd_above_signal",
]

CRYPTO_50 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "UNIUSDT", "AAVEUSDT", "LTCUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "FILUSDT",
    "INJUSDT", "FETUSDT", "RENDERUSDT", "TIAUSDT", "JUPUSDT",
    "WUSDT", "PENDLEUSDT", "ENAUSDT", "ATOMUSDT", "ALGOUSDT",
    "FTMUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "GALAUSDT",
    "ICPUSDT", "VETUSDT", "EGLDUSDT", "THETAUSDT", "HBARUSDT",
    "LDOUSDT", "MKRUSDT", "SNXUSDT", "COMPUSDT", "GRTUSDT",
    "STXUSDT", "IMXUSDT", "RUNEUSDT", "CFXUSDT", "SEIUSDT",
]

STOCKS_100 = [
    "SPY", "QQQ", "IWM", "DIA", "NVDA", "TSLA", "AAPL", "MSFT",
    "AMZN", "GOOGL", "META", "AMD", "COIN", "MSTR", "PLTR",
    "SOFI", "PYPL", "MARA", "RIOT", "ARM", "SMCI", "INTC",
    "NFLX", "CRM", "UBER", "SHOP", "RBLX", "SNOW", "NET",
    "DDOG", "MDB", "CRWD", "ZS", "PANW", "ABNB", "DASH",
    "BA", "JPM", "GS", "V", "MA", "UNH", "JNJ", "PFE",
    "LLY", "ABBV", "XOM", "CVX", "COP", "GLD", "SLV",
    "XLF", "XLE", "XLK", "ARKK", "IBIT", "TQQQ", "SOXL",
    "DIS", "WMT", "COST", "HD", "TGT", "SBUX", "MCD",
    "KO", "PEP", "PG", "NKE", "LULU", "F", "GM",
    "DE", "CAT", "RTX", "LMT",
]


async def fetch_klines(symbol, interval, days, client):
    """Fetch all klines with pagination."""
    all_data = []
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 86400000)
    current = start_time

    while current < end_time:
        try:
            resp = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": symbol, "interval": interval,
                        "startTime": str(current), "limit": "1000"},
            )
            if resp.status_code == 429:
                await asyncio.sleep(3)
                continue
            data = resp.json()
            if not data or not isinstance(data, list):
                break
            for k in data:
                all_data.append([float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
            current = data[-1][6] + 1
            if len(data) < 1000:
                break
            await asyncio.sleep(0.05)
        except Exception:
            break

    return pd.DataFrame(all_data, columns=["open", "high", "low", "close", "volume"]) if all_data else pd.DataFrame()


def fast_extract(df, bar_idx, direction, fear_greed=50, funding=0.0):
    """Extract features without heavy imports each time."""
    import ta
    feats = extract_features(df, bar_idx, direction)
    feats["bias_is_bullish"] = 1.0 if direction == "long" else 0.0
    feats["funding_rate"] = funding * 10000
    feats["fear_greed"] = float(fear_greed)
    feats["higher_tf_aligned"] = 0.0
    feats["btc_trend_bullish"] = 0.5
    feats["hour_sin"] = np.sin(2 * np.pi * (bar_idx % 24) / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * (bar_idx % 24) / 24)
    feats["day_of_week"] = float(bar_idx % 7)

    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    if bar_idx >= 50 and bar_idx < len(df):
        v = float(volume.iloc[bar_idx])
        v5 = float(volume.iloc[max(0,bar_idx-5):bar_idx+1].mean())
        v50 = float(volume.iloc[max(0,bar_idx-50):bar_idx+1].mean())
        feats["vol_ratio_5"] = v/v5 if v5>0 else 1
        feats["vol_ratio_50"] = v/v50 if v50>0 else 1
        h20 = float(high.iloc[max(0,bar_idx-20):bar_idx+1].max())
        l20 = float(low.iloc[max(0,bar_idx-20):bar_idx+1].min())
        rng = h20-l20
        feats["range_position"] = (float(close.iloc[bar_idx])-l20)/rng if rng>0 else 0.5
        c,o,h_,l_ = float(close.iloc[bar_idx]),float(open_.iloc[bar_idx]),float(high.iloc[bar_idx]),float(low.iloc[bar_idx])
        body = abs(c-o)
        feats["body_upper_wick_ratio"] = (h_-max(c,o))/body if body>0 else 0
        feats["body_lower_wick_ratio"] = (min(c,o)-l_)/body if body>0 else 0
        rsi = ta.momentum.RSIIndicator(close.iloc[:bar_idx+1], window=14).rsi()
        feats["rsi_slope"] = float(rsi.iloc[-1])-float(rsi.iloc[-4]) if len(rsi)>=4 else 0
        mi = ta.trend.MACD(close.iloc[:bar_idx+1])
        ml,sl = mi.macd(),mi.macd_signal()
        feats["macd_above_signal"] = 1.0 if len(ml)>0 and len(sl)>0 and float(ml.iloc[-1])>float(sl.iloc[-1]) else 0
    else:
        for f in ["vol_ratio_5","vol_ratio_50","range_position","body_upper_wick_ratio",
                   "body_lower_wick_ratio","rsi_slope","macd_above_signal"]:
            feats[f] = 0.0
    return feats


def process_df_chunked(df, symbol, interval, max_hold, fear_greed=50, funding=0.0):
    """Process a large DataFrame by splitting into overlapping 300-bar chunks."""
    trades_data = []
    chunk_size = 300
    step = 200  # overlap of 100 bars

    for start in range(0, max(1, len(df) - chunk_size), step):
        chunk = df.iloc[start:start + chunk_size].reset_index(drop=True)
        if len(chunk) < 100:
            continue

        result = backtest_df(chunk, symbol, interval, min_confidence=0.4,
                            lookback_window=50, max_hold_bars=max_hold)

        for trade in result.trades:
            if trade.setup_type not in ("ob_retest", "fvg_fill"):
                continue
            feats = fast_extract(chunk, trade.entry_bar, trade.direction, fear_greed, funding)
            label = 1 if trade.pnl_pct > 0 else 0
            trades_data.append((feats, label, trade.pnl_pct))

    return trades_data


async def main():
    t0 = time.time()
    print("=" * 65)
    print("  PHASE 1.1 FAST — Chunked Historical Pipeline")
    print("=" * 65)
    sys.stdout.flush()

    from market_scanner.providers.scrapers import scrape_fear_greed, scrape_funding_rates
    fg = await scrape_fear_greed()
    fear_greed = fg.value if fg else 50
    funding = await scrape_funding_rates()
    funding_map = {r.symbol: r.rate for r in funding}
    print(f"Context: fear_greed={fear_greed} funding_pairs={len(funding_map)}")
    sys.stdout.flush()

    all_f, all_l, all_p = [], [], []

    async with httpx.AsyncClient(timeout=30) as client:
        # ── Crypto 1h: 50 pairs x 2 years ────────────────────────────
        print(f"\nPhase A: Crypto 1h (50 pairs x 2yr)")
        sys.stdout.flush()
        for i, sym in enumerate(CRYPTO_50):
            df = await fetch_klines(sym, "1h", 730, client)
            if len(df) < 200:
                continue
            fr = funding_map.get(sym, 0)
            rows = process_df_chunked(df, sym, "1h", 48, fear_greed, fr)
            for f, l, p in rows:
                all_f.append(f); all_l.append(l); all_p.append(p)
            print(f"  [{i+1:>2}/50] {sym}: {len(df):>6} bars -> {len(rows):>3} trades  (total={len(all_f)})")
            sys.stdout.flush()

        c1 = len(all_f)
        print(f"  Subtotal: {c1} trades")
        sys.stdout.flush()

        # ── Crypto 4h: 50 pairs x 2 years ────────────────────────────
        print(f"\nPhase B: Crypto 4h (50 pairs x 2yr)")
        sys.stdout.flush()
        for i, sym in enumerate(CRYPTO_50):
            df = await fetch_klines(sym, "4h", 730, client)
            if len(df) < 200:
                continue
            fr = funding_map.get(sym, 0)
            rows = process_df_chunked(df, sym, "4h", 24, fear_greed, fr)
            for f, l, p in rows:
                all_f.append(f); all_l.append(l); all_p.append(p)
            if (i+1) % 10 == 0:
                print(f"  [{i+1:>2}/50] total={len(all_f)}")
                sys.stdout.flush()

        c2 = len(all_f) - c1
        print(f"  Subtotal: {c2} trades")
        sys.stdout.flush()

        # ── Crypto 15m: 20 pairs x 6 months ──────────────────────────
        print(f"\nPhase C: Crypto 15m (20 pairs x 6mo)")
        sys.stdout.flush()
        for i, sym in enumerate(CRYPTO_50[:20]):
            df = await fetch_klines(sym, "15m", 180, client)
            if len(df) < 200:
                continue
            rows = process_df_chunked(df, sym, "15m", 96, fear_greed)
            for f, l, p in rows:
                all_f.append(f); all_l.append(l); all_p.append(p)
            if (i+1) % 5 == 0:
                print(f"  [{i+1:>2}/20] total={len(all_f)}")
                sys.stdout.flush()

    c3 = len(all_f) - c1 - c2
    print(f"  Subtotal: {c3} trades")
    sys.stdout.flush()

    # ── Stocks daily: 75 tickers x 5 years ────────────────────────────
    print(f"\nPhase D: Stocks daily (75 tickers x 5yr)")
    sys.stdout.flush()
    import yfinance as yf
    for i, ticker in enumerate(STOCKS_100[:75]):
        try:
            df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            if df.empty or len(df) < 200:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df = df[["open","high","low","close","volume"]].dropna().reset_index(drop=True)

            rows = process_df_chunked(df, ticker, "1d", 20, fear_greed)
            for f, l, p in rows:
                all_f.append(f); all_l.append(l); all_p.append(p)
            if (i+1) % 15 == 0:
                print(f"  [{i+1:>2}/75] total={len(all_f)}")
                sys.stdout.flush()
        except Exception:
            pass

    c4 = len(all_f) - c1 - c2 - c3
    total = len(all_f)
    wins = sum(all_l)
    elapsed = time.time() - t0

    print(f"\n{'='*65}")
    print(f"DATA: {total} trades in {elapsed:.0f}s")
    print(f"  1h={c1}  4h={c2}  15m={c3}  stocks={c4}")
    print(f"  Wins={wins} Losses={total-wins} WR={wins/total:.1%}")
    print(f"{'='*65}")
    sys.stdout.flush()

    if total < 500:
        print("Not enough data. Exiting.")
        return

    mdir = Path(__file__).parent / ".." / "programs" / "market-scanner" / "models"
    mdir.mkdir(exist_ok=True)

    X = np.array([[r.get(n, 0) for n in V2_FEATURES] for r in all_f])
    y = np.array(all_l)
    pnls = np.array(all_p)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Save raw data so we never have to re-collect
    data_path = mdir / "v3_training_data.npz"
    np.savez_compressed(data_path, X=X, y=y, pnls=pnls)
    print(f"\nTraining data saved: {data_path.name} ({X.shape})")
    sys.stdout.flush()

    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    # Try GPU, fall back to CPU
    try:
        test_model = xgb.XGBClassifier(device="cuda", n_estimators=10)
        test_model.fit(X[:100], y[:100])
        device = "cuda"
        print(f"\nTraining v3 on GPU (RTX 4060) — {total} samples, {len(V2_FEATURES)} features")
    except Exception:
        device = "cpu"
        print(f"\nTraining v3 on CPU — {total} samples, {len(V2_FEATURES)} features")
    sys.stdout.flush()

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, min_child_weight=20, random_state=42,
        device=device, eval_metric="logloss",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    prec = cross_val_score(model, X, y, cv=cv, scoring="precision")
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")

    print(f"\nV3 RESULTS ({device.upper()})")
    print(f"  Accuracy:  {np.mean(acc):.1%} (+/- {np.std(acc):.1%})")
    print(f"  Precision: {np.mean(prec):.1%} (+/- {np.std(prec):.1%})")
    print(f"  F1:        {np.mean(f1):.1%} (+/- {np.std(f1):.1%})")
    sys.stdout.flush()

    model.fit(X, y)
    imps = sorted(zip(V2_FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\nTOP FEATURES:")
    for feat, imp in imps[:10]:
        print(f"  {feat:<30} {imp:.4f} {'#'*int(imp*200)}")
    sys.stdout.flush()

    # Save
    name = f"v3_{total}trades_{np.mean(acc)*100:.0f}pct.pkl"
    with open(mdir / name, "wb") as f:
        pickle.dump({"model": model, "features": V2_FEATURES, "version": 3, "trades": total}, f)
    with open(mdir / "v3_latest.pkl", "wb") as f:
        pickle.dump({"model": model, "features": V2_FEATURES, "version": 3, "trades": total}, f)
    print(f"\nSaved: {name}")

    # Sim
    probas = model.predict_proba(X)[:, 1]
    print(f"\n{'='*65}")
    print(f"  {'Filter':<12} {'Trades':>7} {'WR':>8} {'Avg':>8} {'Total PnL':>12}")
    print(f"  {'-'*50}")
    print(f"  {'raw':<12} {len(y):>7} {y.mean():>8.1%} {pnls.mean():>8.2f}% {pnls.sum():>11.1f}%")
    for t in [.50,.55,.60,.65,.70]:
        m = probas >= t
        if m.sum() < 5: continue
        print(f"  >={t:<10.0%} {m.sum():>7} {y[m].mean():>8.1%} {pnls[m].mean():>8.2f}% {pnls[m].sum():>11.1f}%")
    print(f"{'='*65}")
    print(f"Total time: {time.time()-t0:.0f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
