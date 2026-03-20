# ORACLE — PRD Source

**Source:** Inline PRD pasted by user on 2026-03-14
**Version:** v0.1 DRAFT
**Do not delete** — permanent record of what initiated the scaffold.

---

# ORACLE — Open Reasoning and Compounding Logic for Edge
## Product Requirements Document · v0.1 · DRAFT

[Full PRD content preserved below]

## Abstract

ORACLE is a modular, AI-native trading intelligence platform targeting prediction markets and on-chain asset markets. It operates as a continuously-running reasoning system — not a simple rules engine — capable of ingesting multi-modal signal streams, synthesizing them into trade theses, executing positions, and generating post-mortem intelligence from every closed trade. The system is designed to compound its own edge over time through a self-curating knowledge base of markdown documents and an embedded reasoning loop that learns from both wins and losses.

This document defines *what* ORACLE must do, not how it is built. No specific languages, frameworks, or infrastructure providers are prescribed.

## 1. Problem Space

Polymarket and on-chain asset markets are deeply inefficient at the edges. Retail participants are slow, emotionally reactive, and information-poor. Large sophisticated actors move first — often telegraphing their intentions on-chain through outsized buys that ripple into price before the broader market reprices. Meanwhile, open-source intelligence (OSINT) signals are publicly accessible but rarely synthesized fast enough to be actionable.

A system that can simultaneously monitor whale wallet activity on Polymarket, semantically fuse OSINT signals, and run a reasoning layer that continuously asks "what is the highest expected value trade available right now?" would have a structural edge over the median participant.

Additionally, Solana-native token markets present a complementary opportunity: volatile assets with predictable mean-reverting behavior that can be exploited with a simple but AI-calibrated buy-the-dip / take-profit loop.

## 2. Core System Modules

Six loosely coupled modules communicating through a shared internal event bus and shared state layer.

### 2.1 Signal Ingestion Layer (SIL)
Sources: Polymarket REST & WebSocket API, Polymarket on-chain CLOB events (OrderFilled, OrderPlaced on Polygon), OSINT feeds (news aggregators, government portals, sports APIs, social velocity, Wikipedia, prediction subreddits), Solana price oracles, AI opinion streams.
Every Signal carries: source_id, timestamp, category (on-chain/news/social/price/ai-generated), raw_payload, confidence.

### 2.2 Whale & Anomaly Detection Engine (WADE)
Threshold-based flagging (configurable USD notional). Wallet intelligence tagging with persistent registry (historical PnL, win rate, position size, category preference, reputation tier: Shark/Informed/Unknown/Noise). Pattern recognition for cascading buys. Anomaly scoring. Copy-trade interface with three operator modes.

### 2.3 OSINT Semantic Fusion Engine (OSFE)
Vector embedding of every incoming signal. Semantic similarity search against active market registry (probabilistic, not keyword-based). Continuous semantic state per tracked market. Source credibility weighting (configurable, updated by post-mortems).

### 2.4 Solana Opportunistic Executor (SOE)
Autonomous mean-reversion strategy. Statistical model: N-day MA, std dev bands, price velocity, AI floor estimate (from RE). Entry: price < MA AND within AI floor zone. Exit: take-profit (fixed % or dynamic) or stop-loss. Risk controls: max concurrent positions, max capital per position, daily loss circuit breaker.

### 2.5 Reasoning Engine (RE)
Multi-pass pipeline: Step 1 — context assembly (MarketState + whale activity + historical analogues from KBPM). Step 2 — adversarial hypothesis generation (model argues both sides). Step 3 — evidence weighting + probability estimate vs market implied probability. Step 4 — confidence calibration. Scheduled full market scan (configurable interval, e.g. 30 min).

### 2.6 Knowledge Base & Post-Mortem System (KBPM)
Markdown vault: /markets/, /wallets/, /signals/, /theses/, /osint/. Every thesis documented (executed or not). Post-mortem on resolution: signals present, thesis, what happened, counterfactual, source weight updates. KBPM feeds back to RE via semantic search over /theses/.

## 3. Operator Interface
Live anomaly alerts, active theses with confidence, SOE positions with PnL, post-mortem feed, parameter control panel. Copy-trade modes: fully manual / semi-automatic / fully automatic.

## 4. Non-Functional Requirements
Latency: on-chain fill → copy-trade decision < 10 seconds. Resilience: individual module failures don't cause total shutdown. Auditability: every decision logged with full reasoning context. Capital safety: hard-coded circuit breakers at infrastructure level. Modularity: each module is replaceable without changing others.

## 5. Success Metrics
(a) RE theses demonstrate positive EV over 30-day rolling window. (b) WADE copy-trades on Tier 1 wallets outperform random entry at statistically significant rates. (c) SOE achieves target monthly returns without triggering circuit breakers. (d) Post-mortems produce measurable improvements in signal weighting over time.
