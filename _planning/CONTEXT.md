# _planning — Architecture Workspace

## What This Is
Where decisions get made before code gets written.
Do not create files in programs/ without checking whether a decision exists here first.

---

## What Lives Here

| Folder/File | Purpose |
|-------------|---------|
| adr/ | Architecture Decision Records — 22 total, 10 accepted, 12 assumption |
| system-design/ | Data flows, service boundaries, component diagrams |
| roadmap.md | Build order — what gets built first and why |
| prd-source.md | Original PRD — permanent record, never delete |

---

## Task Routing

| Your Task | Do This |
|-----------|---------|
| Make a new decision | Check adr/ first. If it exists, read it. If not, write a new ADR. |
| Validate an assumption ADR | Read the ADR, make the decision, change status to `accepted`, close the linked gap JSON |
| Document system design | Create or update files in system-design/ |
| Check build order | Read roadmap.md |

---

## ADR Status Summary

### Accepted (from PRD — binding now)
| ADR | Decision |
|-----|----------|
| ADR-001 | Modular event-bus architecture — 6 core modules + dashboard |
| ADR-002 | No language/framework prescription — implementation can evolve |
| ADR-003 | OSFE uses vector embeddings for semantic market association |
| ADR-004 | Reasoning engine uses multi-pass adversarial reasoning, not single prompt |
| ADR-005 | Knowledge base is a file-system markdown vault |
| ADR-006 | Capital safety circuit breakers enforced at infrastructure level |
| ADR-007 | On-chain fill → copy-trade decision latency must be under 10 seconds |
| ADR-008 | Every decision (executed or not) logged with full reasoning context |
| ADR-009 | SOE operates independently but shares RE and KBPM |
| ADR-010 | Copy-trade has three modes: fully manual, semi-automatic, fully automatic |

### All ADRs Accepted — No Blockers Remain
| ADR | Decision |
|-----|----------|
| ADR-011 | Alchemy polygon-mainnet WebSocket, web3.py AsyncWeb3 |
| ADR-012 | OpenAI text-embedding-3-small, 512 dims |
| ADR-013 | ChromaDB local, collections: oracle_markets + oracle_theses |
| ADR-014 | Redis pub/sub, 8 channels, redis-py async |
| ADR-015 | Redis hash namespace oracle:state:, AOF persistence |
| ADR-016 | Birdeye API (price) + Jupiter v6 (execution) |
| ADR-017 | Server-side env-var keypair, py-clob-client |
| ADR-018 | FastAPI + vanilla HTML/JS, localhost:8080 |
| ADR-019 | Telegram bot, python-telegram-bot async |
| ADR-020 | APScheduler AsyncIOScheduler, default 30min |
| ADR-021 | Separate Solana keypair, SOLANA_PRIVATE_KEY env var |
| ADR-022 | NewsAPI + Wikipedia + Reddit + Polymarket descriptions |
| ADR-023 | Python 3.11+, asyncio, uv, Pydantic v2, Claude claude-sonnet-4-6 |

---

## ADR Format

File: `adr/[NNN]-[decision-slug].md`

Status values:
- `proposed` — written but not yet validated
- `accepted` — binding for all programs in this project
- `assumption` — inferred to fill a PRD gap — needs human validation before the affected program builds
- `superseded by ADR-[NNN]` — replaced, never deleted

An agent encountering an `assumption` ADR must not build the affected feature until
a human changes the status to `accepted`.
