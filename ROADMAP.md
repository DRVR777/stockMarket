# ORACLE — Long-Term Roadmap

## Current State (v0.1 — March 2026)

**What works:**
- 8 programs built, 47 tests passing
- Live scanning: 80,000+ assets across crypto + stocks + 7 EVM chains
- SMC pattern detection: FVG, order blocks, BOS/CHoCH, liquidity sweeps
- ML classifier v2: 64.4% accuracy on 3,186 trades, 33 features
- Web scrapers: fear/greed, funding rates, Reddit sentiment, insider trades
- AI reasoning: Gemini-powered adversarial hypothesis generation (free)
- Infrastructure: Redis + Postgres + ChromaDB live
- Paper trading with circuit breakers
- Chain-agnostic execution (Solana + EVM adapter)

**What doesn't work yet:**
- No live trading (paper only)
- ML model needs more data (3k trades → want 50k+)
- No order book depth data
- No backtester for stocks (only forward-tested)
- Dashboard not stress-tested with live data
- No mobile alerts beyond Telegram

---

## Phase 1: Data Dominance (Weeks 1-4)

**Goal:** 50,000+ training trades, 70%+ ML accuracy

### 1.1 Historical data pipeline
- [ ] Binance full history: pull 2 years of 1h klines for top 50 pairs (876,000 candles)
- [ ] Binance 15m history: 6 months for top 30 pairs (525,600 candles)
- [ ] Yahoo Finance: 5 years daily for top 100 stocks (125,000 bars)
- [ ] Store all raw OHLCV in Postgres (new `candles` table, partitioned by symbol)
- [ ] Batch backtest: run SMC on all historical data, extract 50k+ trades

### 1.2 Order book depth
- [ ] Binance WS depth stream (free): top 20 bid/ask levels for each symbol
- [ ] New features: bid-ask imbalance, wall detection, spoofing patterns
- [ ] Order flow imbalance: delta (buy volume - sell volume) per bar

### 1.3 On-chain analytics
- [ ] Ethereum mempool monitoring: pending large transactions
- [ ] Whale wallet tracking: top 100 wallets per chain, alert on movements
- [ ] DEX volume spikes: detect unusual trading on Uniswap/Raydium before CEX moves
- [ ] Token unlock schedules: scrape from Token Unlocks, factor into model

### 1.4 Alternative data
- [ ] Twitter/X sentiment: trending $tickers, engagement velocity
- [ ] GitHub activity: commit frequency for crypto project repos (dev activity = bullish)
- [ ] Google Trends: search volume for crypto/stock names
- [ ] Options flow: put/call ratio from CBOE (free delayed data)
- [ ] Dark pool prints: FINRA short volume data (free, delayed)

### 1.5 ML model v3
- [ ] Train on 50k+ trades with 50+ features
- [ ] Separate models per asset class (crypto vs stocks) and timeframe
- [ ] Add LSTM/transformer model for sequence prediction alongside GBM
- [ ] Ensemble: combine GBM + LSTM predictions
- [ ] Target: 70%+ accuracy, 1.5+ profit factor in backtest

---

## Phase 2: Execution (Weeks 5-8)

**Goal:** Live paper trading → validated edge → first real trades

### 2.1 Paper trading validation
- [ ] Run full pipeline live for 2 weeks (all components)
- [ ] Track: signals generated, trades filtered, paper PnL
- [ ] Compare ML predictions vs actual outcomes
- [ ] Minimum bar: 60%+ live win rate, positive PnL over 2 weeks

### 2.2 Live execution — crypto
- [ ] SOE live mode: Solana via Jupiter (real transactions)
- [ ] Start with $100 max position, 1 concurrent trade
- [ ] Gradually increase: $100 → $250 → $500 → $1000
- [ ] Circuit breaker: $50/day loss ceiling initially

### 2.3 Live execution — Polymarket
- [ ] Wire RE thesis → Polymarket CLOB execution via py-clob-client
- [ ] Copy-trade path: whale alert → operator approval → auto-execute
- [ ] Start with $50 per trade, manual approval only

### 2.4 Multi-chain execution
- [ ] EVM live mode: Base (lowest gas), then Arbitrum, then Ethereum
- [ ] Implement actual swap signing via web3.py + private key
- [ ] Per-chain circuit breakers and position limits

---

## Phase 3: Intelligence (Weeks 9-16)

**Goal:** Self-improving system that gets smarter every day

### 3.1 Post-mortem feedback loop (live)
- [ ] Every closed trade generates a post-mortem via LLM
- [ ] Source weight adjustments flow back to OSFE credibility weights
- [ ] ML model retrained weekly on new trade data (automated)
- [ ] Track: which features are gaining/losing importance over time

### 3.2 Market regime detection
- [ ] Classify current regime: trending, ranging, volatile, calm
- [ ] Different model weights per regime
- [ ] Auto-adjust: wider stops in volatile regime, tighter in calm
- [ ] Reduce position sizing in high-uncertainty regimes

### 3.3 Correlation engine
- [ ] Cross-asset correlation matrix (updated hourly)
- [ ] BTC leads altcoins by ~15min — exploit the lag
- [ ] SPY/QQQ correlation with crypto during US market hours
- [ ] Sector rotation detection: money flowing from tech → crypto or vice versa

### 3.4 News reaction model
- [ ] Train on: news headline → price move in next 1h/4h/1d
- [ ] Classify news sentiment: bullish/bearish/neutral
- [ ] Speed matters: first to react to breaking news = alpha
- [ ] Wire into RE: news signal boosts/reduces thesis confidence

---

## Phase 4: Scale (Weeks 17-24)

**Goal:** Multi-strategy, multi-market, multi-timeframe

### 4.1 Additional strategies
- [ ] Momentum: ride trends with trailing stops (not just mean reversion)
- [ ] Arbitrage: cross-exchange price differences (Binance vs Coinbase)
- [ ] Funding rate farming: long/short based on funding rate extremes
- [ ] Liquidation cascade: detect liquidation clusters, trade the bounce
- [ ] Earnings plays: buy/sell stocks before earnings based on historical patterns

### 4.2 Portfolio management
- [ ] Portfolio-level risk: max drawdown across all strategies
- [ ] Asset allocation: Kelly criterion for position sizing
- [ ] Hedging: auto-hedge long crypto with short BTC futures in bearish regime
- [ ] Rebalancing: shift capital between strategies based on recent performance

### 4.3 Infrastructure hardening
- [ ] Kubernetes deployment (auto-restart, scaling)
- [ ] Grafana dashboards (latency, PnL, signal throughput)
- [ ] Automated ML retraining pipeline (weekly)
- [ ] Backup: Redis persistence + Postgres replication
- [ ] Multi-region: East + West coast for lower latency to exchanges

### 4.4 Mobile app
- [ ] React Native or Flutter app
- [ ] Push notifications for: trade entry, exit, circuit breaker, whale alerts
- [ ] One-tap approve/reject for copy trades
- [ ] Live PnL tracking

---

## Phase 5: Edge (Months 6-12)

**Goal:** Sustainable alpha that compounds

### 5.1 Alpha research
- [ ] Systematic factor research: test 100+ features for predictive power
- [ ] Walk-forward optimization: prevent overfitting
- [ ] Out-of-sample validation on every strategy change
- [ ] Alpha decay tracking: when does a signal stop working?

### 5.2 Advanced ML
- [ ] Reinforcement learning: agent learns optimal entry/exit timing
- [ ] Graph neural networks: model cross-asset relationships
- [ ] Attention-based models: learn which features matter in different regimes
- [ ] Fine-tuned LLM: train on 1000+ post-mortems for market reasoning

### 5.3 Speed
- [ ] Co-location with exchange servers (if profitable enough to justify cost)
- [ ] Rust execution layer for sub-millisecond order placement
- [ ] Pre-computed order blocks: don't recalculate, just check
- [ ] WebSocket multiplexing: single connection for all data streams

### 5.4 New markets
- [ ] Forex: EUR/USD, GBP/USD via OANDA (free practice account)
- [ ] Commodities: gold, oil futures (via Yahoo Finance or free APIs)
- [ ] Sports prediction markets (Polymarket, Kalshi)
- [ ] Election markets: long-term thesis trading

---

## Success Metrics

| Timeframe | Metric | Target |
|-----------|--------|--------|
| Month 1 | ML accuracy (cross-validated) | 70%+ |
| Month 1 | Historical backtest profit factor | 1.5+ |
| Month 2 | Live paper trading win rate | 60%+ |
| Month 2 | Live paper PnL (2 weeks) | Positive |
| Month 3 | First real trade executed | ✓ |
| Month 3 | Live win rate | 55%+ |
| Month 6 | Monthly return (risk-adjusted) | 5-10% |
| Month 6 | Max drawdown | <15% |
| Year 1 | Strategies running | 5+ |
| Year 1 | Assets monitored | 1000+ actively |
| Year 1 | Annual return | 50-100% target |

---

## Principles

1. **Data before trades.** Never trade a strategy that hasn't been backtested on 6+ months of data.
2. **Paper before live.** 2 weeks of paper trading minimum before real money.
3. **Small before big.** Start at $100 per trade, scale up only after proven edge.
4. **ML before human.** Trust the model's filter over gut feeling. If ML says skip, skip.
5. **Survive before thrive.** Circuit breakers, position limits, daily loss ceilings — always on.
6. **Free before paid.** Maximize free APIs before paying for data. Most alpha is in free data.
7. **Compound the edge.** Every post-mortem makes the model smarter. Every week of data improves predictions.
