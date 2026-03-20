# Gap Registry — oracle

## Scope: oracle-internal only
Cross-project gaps go to root _meta/gaps/.

## Current Open Gaps

None — all 12 intake gaps closed. See Closed Gaps below.

## Closed Gaps

| Gap ID | ADR | Description | Status | Closed By |
|--------|-----|-------------|--------|-----------|
| gap-001 | ADR-011 | Polygon RPC provider | closed | ADR-011 accepted — Alchemy, web3.py AsyncWeb3 |
| gap-002 | ADR-012 | Embedding model | closed | ADR-012 accepted — OpenAI text-embedding-3-small, 512 dims |
| gap-003 | ADR-013 | Vector store | closed | ADR-013 accepted — ChromaDB, 2 collections: oracle_markets + oracle_theses |
| gap-004 | ADR-014 | Event bus | closed | ADR-014 accepted — Redis pub/sub, 8 channels, redis-py async |
| gap-005 | ADR-015 | Shared state layer | closed | ADR-015 accepted — Redis hash namespace oracle:state:, AOF persistence |
| gap-006 | ADR-016 | Solana price oracle + DEX | closed | ADR-016 accepted — Birdeye API + Jupiter v6 |
| gap-007 | ADR-017 | Polymarket wallet management | closed | ADR-017 accepted — env-var keypair, py-clob-client |
| gap-008 | ADR-018 | Operator dashboard delivery | closed | ADR-018 accepted — FastAPI + vanilla HTML/JS, localhost:8080 |
| gap-009 | ADR-019 | Alert notification delivery | closed | ADR-019 accepted — Telegram bot, python-telegram-bot |
| gap-010 | ADR-020 | RE scheduled scan trigger | closed | ADR-020 accepted — APScheduler AsyncIOScheduler, default 30min |
| gap-011 | ADR-021 | SOE wallet separation | closed | ADR-021 accepted — separate Solana keypair, SOLANA_PRIVATE_KEY env var |
| gap-012 | ADR-022 | OSINT source launch list | closed | ADR-022 accepted — NewsAPI + Wikipedia + Reddit + Polymarket descriptions |

## Status
All gaps closed. All ADRs accepted (ADR-001 through ADR-023).
All 9 contracts defined (no stubs remaining).

**Every program is now unblocked for spec-review.**

## Next Step
Run spec-review on each program in roadmap order:
1. signal-ingestion
2. whale-detector + osint-fusion (parallel)
3. reasoning-engine
4. knowledge-base + solana-executor (parallel)
5. operator-dashboard

See `../../_planning/roadmap.md` for build order rationale.
