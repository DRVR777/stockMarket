# MANIFEST — programs/signal-ingestion

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-programs-signal-ingestion |
| `type` | program |
| `depth` | 4 |
| `parent` | oracle/programs/ |
| `status` | active |

## What I Am
The Signal Ingestion Layer (SIL). Continuously polls and subscribes to all data sources — Polymarket REST/WS API, Polygon CLOB contract events, OSINT feeds, Solana price oracles, and AI opinion streams — and normalizes everything into canonical Signal objects published to the event bus.

## External Dependencies
| Depends On | What | Contract Location |
|------------|------|-------------------|
| event bus (ADR-014) | publish Signal objects | ../../shared/contracts/signal.md |
| Polymarket REST+WS API | active markets, orderbook, trade history | external |
| Polygon RPC — Alchemy (ADR-011) | OrderFilled, OrderPlaced on-chain events | external |
| OSINT sources — NewsAPI, Wikipedia, Reddit (ADR-022) | news, social feeds | external |
| Solana price oracle — Birdeye (ADR-016) | Solana asset prices via WS | external |

## Gap Status
All gaps closed. All ADRs accepted. Spec-review PASS — 2026-03-14.

## What I Produce
Signal objects on the `oracle:signal` Redis channel. One Signal per discrete data event, normalized to the shape in ../../shared/contracts/signal.md.

## Routing Rules
| Condition | Go To |
|-----------|-------|
| Build an adapter | CONTEXT.md — follow build sequence row by row |
| Architecture question | ../../_planning/CONTEXT.md |
| Update Signal shape | ../../shared/contracts/signal.md first, then return here |
