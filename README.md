# ORACLE — AI Trading Intelligence Platform

AI-native trading intelligence system that scans crypto and stock markets, detects institutional order flow patterns, generates adversarial trade theses, and executes mean-reversion strategies.

## Architecture

```
[Market Scanner] — scans 3000+ crypto + stocks for SMC patterns (FVG, OB, BOS, liquidity)
       |
[Signal Ingestion] — 8 data adapters (Polymarket, Polygon, NewsAPI, Reddit, Birdeye, etc.)
       |
[Whale Detector] ←——→ [OSINT Fusion]
  anomaly scoring       semantic embedding + ChromaDB similarity
  cascade detection     source credibility weighting
       |                      |
[Reasoning Engine] — 4-step adversarial pipeline via LLM
  context assembly → hypothesis generation → evidence weighting → confidence calibration
       |
[Solana Executor] — chain-agnostic mean-reversion trader (paper mode default)
       |
[Knowledge Base] — markdown vault + post-mortem generation (self-improving feedback loop)
       |
[Operator Dashboard] — FastAPI + vanilla JS at localhost:8080
```

## 8 Programs

| Program | What it does |
|---------|-------------|
| `signal-ingestion` | Polls 8 data sources, normalizes to Signal objects |
| `whale-detector` | Scores on-chain anomalies, tracks wallets, surfaces copy-trade opportunities |
| `osint-fusion` | Embeds signals, matches to markets via ChromaDB, maintains semantic state |
| `reasoning-engine` | Multi-pass adversarial reasoning → TradeThesis with confidence scores |
| `solana-executor` | Chain-agnostic execution via pluggable adapters (Solana first) |
| `knowledge-base` | Markdown vault, thesis indexing, Claude post-mortem generation |
| `operator-dashboard` | Real-time web UI with alerts, theses, positions, parameter control |
| `market-scanner` | Technical analysis + SMC pattern detection across all markets |

## Market Scanner — Smart Money Concepts

Detects institutional patterns across thousands of assets:

- **Fair Value Gaps (FVG)** — imbalance zones where price moves too fast
- **Order Blocks (OB)** — institutional accumulation/distribution zones
- **Break of Structure (BOS)** — trend continuation confirmation
- **Change of Character (CHoCH)** — early reversal detection
- **Liquidity sweeps** — stop-loss hunting above/below swing points
- **Premium/Discount zones** — Fibonacci-based entry zones
- **Displacement** — strong momentum candles confirming institutional activity

## Quick Start

```bash
# 1. Set up infrastructure
cp .env.example .env          # add your GEMINI_API_KEY (free)
docker compose up -d          # Redis + Postgres
python scripts/db_init.py     # create database tables

# 2. Run the scanner (no API keys needed)
cd programs/market-scanner
python -m market_scanner

# 3. Run the full pipeline
make signal-ingestion         # start data ingestion
make whale-detector           # anomaly detection
make osint-fusion             # semantic analysis
make reasoning-engine         # AI reasoning
make solana-executor          # paper trading
make knowledge-base           # vault + post-mortems
make operator-dashboard       # http://localhost:8080
```

## AI Provider Support

Pluggable — switch between providers via env var:

| Provider | Cost | Set in .env |
|----------|------|-------------|
| **Gemini** (default) | Free | `GEMINI_API_KEY=...` |
| Anthropic Claude | Pay-as-you-go | `ANTHROPIC_API_KEY=...` |
| OpenAI | Pay-as-you-go | `OPENAI_API_KEY=...` |

## Stack

Python 3.11+ • asyncio • Pydantic v2 • SQLAlchemy + asyncpg • Redis pub/sub • PostgreSQL • ChromaDB • FastAPI • Gemini/Claude/OpenAI • yfinance • CoinGecko • Birdeye + Jupiter (Solana)

## Tests

```bash
# 47 tests across all programs
python -m pytest programs/whale-detector/tests/ -v
python -m tests.test_pipeline  # in each program directory
```
