"""Parallel data collection while Phase 1.1 runs.

Pulls everything we can get for free and saves to disk:
1. Binance order book snapshots (bid/ask depth) — new data type
2. Crypto social metrics (CoinGecko community data)
3. Stock short interest + institutional holdings
4. Options unusual activity
5. Economic calendar events
6. GitHub dev activity for top crypto projects
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).parent / ".." / "programs" / "market-scanner" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
}

CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "UNIUSDT", "AAVEUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
]


async def scrape_order_books():
    """Binance order book depth for top pairs — free, no auth."""
    print("\n=== 1. Order Book Depth (Binance) ===")
    all_books = {}
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        for sym in CRYPTO_SYMBOLS:
            try:
                resp = await client.get(
                    f"https://api.binance.com/api/v3/depth",
                    params={"symbol": sym, "limit": 20},
                )
                data = resp.json()
                bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
                asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

                bid_wall = max(bids, key=lambda x: x[1]) if bids else (0, 0)
                ask_wall = max(asks, key=lambda x: x[1]) if asks else (0, 0)
                bid_total = sum(q for _, q in bids)
                ask_total = sum(q for _, q in asks)
                imbalance = (bid_total - ask_total) / (bid_total + ask_total) if (bid_total + ask_total) > 0 else 0

                all_books[sym] = {
                    "bid_wall_price": bid_wall[0],
                    "bid_wall_qty": bid_wall[1],
                    "ask_wall_price": ask_wall[0],
                    "ask_wall_qty": ask_wall[1],
                    "bid_total": bid_total,
                    "ask_total": ask_total,
                    "imbalance": round(imbalance, 4),
                    "spread_pct": round((asks[0][0] - bids[0][0]) / bids[0][0] * 100, 4) if bids and asks else 0,
                }
                bias = "BUY" if imbalance > 0.1 else "SELL" if imbalance < -0.1 else "NEUTRAL"
                print(f"  {sym:<12} imbalance={imbalance:+.3f} ({bias})  spread={all_books[sym]['spread_pct']:.4f}%")
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"  {sym}: {str(e)[:40]}")

    with open(DATA_DIR / "order_books.json", "w") as f:
        json.dump(all_books, f, indent=2)
    print(f"  Saved {len(all_books)} order books")


async def scrape_crypto_social():
    """CoinGecko community + developer data — free, no auth."""
    print("\n=== 2. Crypto Social Metrics (CoinGecko) ===")
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()

    coin_ids = {
        "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
        "BNBUSDT": "binancecoin", "XRPUSDT": "ripple", "ADAUSDT": "cardano",
        "DOGEUSDT": "dogecoin", "AVAXUSDT": "avalanche-2", "LINKUSDT": "chainlink",
        "DOTUSDT": "polkadot", "UNIUSDT": "uniswap", "AAVEUSDT": "aave",
    }

    social_data = {}
    loop = asyncio.get_event_loop()
    for sym, coin_id in coin_ids.items():
        try:
            data = await loop.run_in_executor(None, lambda cid=coin_id: cg.get_coin_by_id(
                cid, localization=False, tickers=False, market_data=False,
                community_data=True, developer_data=True,
            ))
            community = data.get("community_data", {})
            developer = data.get("developer_data", {})
            sentiment = data.get("sentiment_votes_up_percentage", 0)

            social_data[sym] = {
                "twitter_followers": community.get("twitter_followers", 0),
                "reddit_subscribers": community.get("reddit_subscribers", 0),
                "reddit_active_48h": community.get("reddit_accounts_active_48h", 0),
                "telegram_members": community.get("telegram_channel_user_count", 0),
                "github_stars": developer.get("stars", 0),
                "github_forks": developer.get("forks", 0),
                "github_commits_4w": developer.get("commit_count_4_weeks", 0),
                "sentiment_up_pct": sentiment,
            }
            print(f"  {sym:<12} reddit={community.get('reddit_subscribers',0):>8} twitter={community.get('twitter_followers',0):>10} commits_4w={developer.get('commit_count_4_weeks',0)}")
            await asyncio.sleep(1.5)  # CoinGecko rate limit
        except Exception as e:
            print(f"  {sym}: {str(e)[:40]}")

    with open(DATA_DIR / "crypto_social.json", "w") as f:
        json.dump(social_data, f, indent=2)
    print(f"  Saved {len(social_data)} coin social profiles")


async def scrape_open_interest():
    """Binance futures open interest — free, no auth."""
    print("\n=== 3. Futures Open Interest (Binance) ===")
    oi_data = {}
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        for sym in CRYPTO_SYMBOLS:
            try:
                resp = await client.get(
                    "https://fapi.binance.com/fapi/v1/openInterest",
                    params={"symbol": sym},
                )
                data = resp.json()
                oi = float(data.get("openInterest", 0))

                # Get 24h change via ticker
                resp2 = await client.get(
                    "https://fapi.binance.com/fapi/v1/ticker/24hr",
                    params={"symbol": sym},
                )
                ticker = resp2.json()
                price = float(ticker.get("lastPrice", 0))
                volume_24h = float(ticker.get("quoteVolume", 0))

                oi_data[sym] = {
                    "open_interest": oi,
                    "oi_value_usd": oi * price,
                    "volume_24h_usd": volume_24h,
                    "oi_volume_ratio": (oi * price) / volume_24h if volume_24h > 0 else 0,
                    "price": price,
                }
                print(f"  {sym:<12} OI=${oi*price/1e6:>8.1f}M  vol=${volume_24h/1e6:>8.1f}M  ratio={oi_data[sym]['oi_volume_ratio']:.2f}")
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"  {sym}: {str(e)[:40]}")

    with open(DATA_DIR / "open_interest.json", "w") as f:
        json.dump(oi_data, f, indent=2)
    print(f"  Saved {len(oi_data)} OI records")


async def scrape_long_short_ratio():
    """Binance top trader long/short ratio — free, no auth."""
    print("\n=== 4. Long/Short Ratio (Binance) ===")
    ls_data = {}
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        for sym in CRYPTO_SYMBOLS:
            try:
                resp = await client.get(
                    "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
                    params={"symbol": sym, "period": "1h", "limit": 1},
                )
                data = resp.json()
                if data and isinstance(data, list):
                    entry = data[0]
                    ratio = float(entry.get("longShortRatio", 1))
                    long_pct = float(entry.get("longAccount", 0.5)) * 100
                    short_pct = float(entry.get("shortAccount", 0.5)) * 100
                    ls_data[sym] = {
                        "long_short_ratio": ratio,
                        "long_pct": long_pct,
                        "short_pct": short_pct,
                    }
                    bias = "LONG HEAVY" if ratio > 1.5 else "SHORT HEAVY" if ratio < 0.67 else "balanced"
                    print(f"  {sym:<12} L/S={ratio:.2f}  long={long_pct:.0f}%  short={short_pct:.0f}%  ({bias})")
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"  {sym}: {str(e)[:40]}")

    with open(DATA_DIR / "long_short_ratio.json", "w") as f:
        json.dump(ls_data, f, indent=2)
    print(f"  Saved {len(ls_data)} L/S ratios")


async def scrape_stock_short_interest():
    """Stock short interest via Yahoo Finance."""
    print("\n=== 5. Stock Short Interest (Yahoo) ===")
    import yfinance as yf

    tickers = ["TSLA", "NVDA", "AMD", "COIN", "MSTR", "PLTR", "SMCI", "MARA", "RIOT", "SOFI"]
    si_data = {}
    loop = asyncio.get_event_loop()

    for ticker in tickers:
        try:
            info = await loop.run_in_executor(None, lambda t=ticker: yf.Ticker(t).info)
            si_data[ticker] = {
                "short_ratio": info.get("shortRatio", 0),
                "short_pct_float": info.get("shortPercentOfFloat", 0),
                "shares_short": info.get("sharesShort", 0),
                "short_change_pct": info.get("shortPercentOfFloat", 0),
                "institutional_pct": info.get("heldPercentInstitutions", 0),
                "insider_pct": info.get("heldPercentInsiders", 0),
                "float_shares": info.get("floatShares", 0),
            }
            si = si_data[ticker]
            print(f"  {ticker:<8} short_ratio={si['short_ratio']:.1f}  short_float={si['short_pct_float']*100 if si['short_pct_float'] else 0:.1f}%  inst={si['institutional_pct']*100 if si['institutional_pct'] else 0:.0f}%")
        except Exception as e:
            print(f"  {ticker}: {str(e)[:40]}")

    with open(DATA_DIR / "stock_short_interest.json", "w") as f:
        json.dump(si_data, f, indent=2, default=str)
    print(f"  Saved {len(si_data)} stock records")


async def scrape_github_activity():
    """GitHub dev activity for top crypto projects — free, no auth."""
    print("\n=== 6. GitHub Dev Activity ===")
    repos = {
        "BTC": "bitcoin/bitcoin",
        "ETH": "ethereum/go-ethereum",
        "SOL": "solana-labs/solana",
        "DOT": "paritytech/polkadot-sdk",
        "AVAX": "ava-labs/avalanchego",
        "LINK": "smartcontractkit/chainlink",
        "UNI": "Uniswap/v3-core",
        "AAVE": "aave/aave-v3-core",
        "ARB": "OffchainLabs/nitro",
        "OP": "ethereum-optimism/optimism",
    }

    gh_data = {}
    async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
        for coin, repo in repos.items():
            try:
                resp = await client.get(f"https://api.github.com/repos/{repo}")
                data = resp.json()

                # Get recent commit activity
                resp2 = await client.get(f"https://api.github.com/repos/{repo}/stats/commit_activity")
                commits = resp2.json()
                recent_commits = sum(w.get("total", 0) for w in commits[-4:]) if isinstance(commits, list) else 0

                gh_data[coin] = {
                    "stars": data.get("stargazers_count", 0),
                    "forks": data.get("forks_count", 0),
                    "open_issues": data.get("open_issues_count", 0),
                    "commits_4w": recent_commits,
                    "last_push": data.get("pushed_at", ""),
                }
                print(f"  {coin:<6} {repo:<35} stars={data.get('stargazers_count',0):>6} commits_4w={recent_commits:>4}")
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"  {coin}: {str(e)[:40]}")

    with open(DATA_DIR / "github_activity.json", "w") as f:
        json.dump(gh_data, f, indent=2)
    print(f"  Saved {len(gh_data)} repos")


async def main():
    t0 = time.time()
    print("=" * 60)
    print("  PARALLEL DATA COLLECTION (while Phase 1.1 runs)")
    print("=" * 60)

    await scrape_order_books()
    await scrape_open_interest()
    await scrape_long_short_ratio()
    await scrape_crypto_social()
    await scrape_stock_short_interest()
    await scrape_github_activity()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ALL SCRAPERS DONE in {elapsed:.0f}s")

    # List saved files
    files = list(DATA_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files)
    print(f"  Files: {len(files)}  Total: {total_size/1024:.0f}KB")
    for f in files:
        print(f"    {f.name}: {f.stat().st_size/1024:.1f}KB")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
