# ADR-023: Implementation Language and Stack

## Status
accepted

## Context
ADR-002 deferred language choice. With ADRs 011–022 now accepted, the technology choices converge on Python: py-clob-client (Polymarket), solders + solana-py (Solana), web3.py (Polygon), chromadb, redis-py, APScheduler, FastAPI — all have first-class Python support. Using multiple languages would add cross-process IPC complexity with no benefit.

## Decision
**Python 3.11+** across all programs.
- Async runtime: `asyncio` throughout (all I/O is async)
- Package manager: `uv` (fast, modern, replaces pip + venv)
- Each program is a standalone Python package with its own `pyproject.toml`
- Shared utilities (Redis client factory, contract serializers): `oracle-shared` internal package imported by all programs

**Core dependencies (shared):**
- `redis[hiredis]` — event bus + shared state
- `openai` — embeddings (ADR-012)
- `chromadb` — vector store (ADR-013)
- `anthropic` — RE reasoning calls (Claude claude-sonnet-4-6)
- `pydantic` v2 — contract object validation and serialization

**Per-program dependencies:**
- signal-ingestion: `web3`, `py-clob-client`, `httpx`, `praw` (Reddit), `apscheduler`
- whale-detector: `redis[hiredis]`
- osint-fusion: `openai`, `chromadb`, `httpx`
- solana-executor: `solders`, `solana`, `httpx` (Birdeye + Jupiter)
- reasoning-engine: `anthropic`, `chromadb`, `apscheduler`
- knowledge-base: `anthropic`, `chromadb`
- operator-dashboard: `fastapi`, `uvicorn`, `websockets`, `python-telegram-bot`

**AI model for RE and KBPM post-mortems:** Claude claude-sonnet-4-6 (`claude-sonnet-4-6`) via Anthropic SDK.
API key: `ANTHROPIC_API_KEY` env var.

## Consequences
- All programs run as separate Python processes (or Docker containers) — no threading across programs
- Contract objects are Pydantic models defined in the `oracle-shared` package; both producer and consumer import the same model class
- `oracle-shared` is installed as a local editable package in each program's virtual environment

## Alternatives Considered
- Node.js: good WebSocket support, but py-clob-client and solana-py have no JS equivalents of comparable maturity
- Mixed Python/Node: adds cross-process complexity for no benefit
