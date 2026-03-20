# ADR-011: Polygon RPC Provider

## Status
accepted

## Context
SIL and WADE require WebSocket subscriptions to Polygon CLOB contract events (OrderFilled, OrderPlaced) with latency under 10 seconds end-to-end (ADR-007).

## Decision
**Alchemy** — polygon-mainnet endpoint via `web3.py` with WebSocket transport.
- Free tier: 300M compute units/month — sufficient for development and moderate live use
- WebSocket support: native, stable, widely documented
- Python SDK (`web3.py`): first-class support
- Fallback: if Alchemy WebSocket drops, reconnect with exponential backoff; log to pending.txt

API key stored as `ALCHEMY_POLYGON_WS_URL` environment variable. Never in code or version control.

## Consequences
- SIL uses `web3.py` AsyncWeb3 + WebSocket provider for on-chain event subscriptions
- WADE receives pre-parsed on-chain Signal objects from SIL — it never calls Alchemy directly
- Alchemy free tier rate limits must be monitored; upgrade plan if limits are hit

## Alternatives Considered
- QuickNode: comparable quality, less generous free tier
- Infura: solid but slower WebSocket reconnect behavior historically
- Self-hosted node: most reliable, too much ops overhead for initial build
