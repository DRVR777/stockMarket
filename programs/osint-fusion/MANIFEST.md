# MANIFEST — programs/osint-fusion

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-programs-osint-fusion |
| `type` | program |
| `depth` | 4 |
| `parent` | oracle/programs/ |
| `status` | specced |

## What I Am
The OSINT Semantic Fusion Engine (OSFE). Subscribes to Signals, embeds text via OpenAI text-embedding-3-small, runs ChromaDB similarity search against active markets, maintains rolling semantic state per market, emits Insight and MarketState objects. Updates source credibility weights from PostMortem feedback.

## External Dependencies
| Depends On | What | Contract Location |
|------------|------|-------------------|
| signal-ingestion | Signal objects | ../../shared/contracts/signal.md |
| event bus — Redis pub/sub (ADR-014) | subscribe Signals + PostMortems, publish Insights + MarketState | ../../shared/contracts/insight.md, market-state.md, post-mortem.md |
| shared state — Redis (ADR-015) | MarketState store, credibility weight params | ../../shared/contracts/market-state.md |
| OpenAI API (ADR-012) | text-embedding-3-small, 512 dims | external |
| ChromaDB (ADR-013) | oracle_markets collection | external |
| Anthropic API (ADR-023) | claude-haiku-4-5-20251001 for semantic_state_summary | external |

## Gap Status
All gaps closed. All ADRs accepted. Spec-review PASS — 2026-03-14.

## What I Produce
Insight objects on `oracle:insight`. MarketState updates on `oracle:market_state` and in Redis `oracle:state:markets`.

## Routing Rules
| Condition | Go To |
|-----------|-------|
| Build a feature | CONTEXT.md — follow build sequence row by row |
| Architecture question | ../../_planning/CONTEXT.md |
| Update a contract | ../../shared/contracts/ first, then return here |
