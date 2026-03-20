# MANIFEST — programs/reasoning-engine

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-programs-reasoning-engine |
| `type` | program |
| `depth` | 4 |
| `parent` | oracle/programs/ |
| `status` | specced |

## What I Am
The Reasoning Engine (RE). Four-step adversarial reasoning pipeline producing TradeThesis objects. Runs on signal trigger (from Insight) and scheduled full scan (APScheduler, 30min default). Serves SOE floor estimate requests via Redis request/reply.

## External Dependencies
| Depends On | What | Contract Location |
|------------|------|-------------------|
| osint-fusion | Insight objects, MarketState | ../../shared/contracts/insight.md, market-state.md |
| whale-detector | AnomalyEvent (indexed per market) | ../../shared/contracts/anomaly-event.md |
| event bus — Redis pub/sub (ADR-014) | subscribe Insights + AnomalyEvents, publish TradeThesis + floor responses | ../../shared/contracts/trade-thesis.md |
| shared state — Redis (ADR-015) | market registry, anomaly index, thesis index, params | internal |
| ChromaDB (ADR-013) | oracle_theses collection — historical analogue search | external |
| Anthropic API (ADR-023) | claude-sonnet-4-6 for hypothesis generation and floor estimates | external |
| APScheduler (ADR-020) | scheduled full market scan | internal |

## Gap Status
All gaps closed. All ADRs accepted. Spec-review PASS — 2026-03-14.

## What I Produce
TradeThesis objects on `oracle:trade_thesis`. SOE floor estimate responses on `oracle:re_floor_response:{request_id}`.

## Routing Rules
| Condition | Go To |
|-----------|-------|
| Build a feature | CONTEXT.md — follow build sequence row by row |
| Architecture question | ../../_planning/CONTEXT.md |
| Update a contract | ../../shared/contracts/ first, then return here |
