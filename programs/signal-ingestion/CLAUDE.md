# CLAUDE.md — signal-ingestion

## What Is In This Directory

| Name | Purpose |
|------|---------|
| `signal_ingestion/` | Python package: entry point, config, and 8 adapter modules |
| `tests/` | Live integration tests per adapter (FakeRedis, no server needed) |
| `pyproject.toml` | Package metadata and dependencies |
| `CONTEXT.md` | Task router with build sequence |
| `MANIFEST.md` | Structural envelope and dependency declarations |

## Quick Rules For This Directory

- All adapters follow the same interface: `start()`, `stop()`, `_normalize()`, `_publish()`
- Publish to `Signal.CHANNEL` (from oracle_shared contracts), never hardcode channel names
- All config values come from `signal_ingestion.config` — no magic numbers in adapter code
- Use `uv` for package management
- Log every inference to `../../_meta/gaps/pending.txt`

## Cross-References

- Signal contract: `../../shared/contracts/signal.md`
- Python model: `../../oracle-shared/oracle_shared/contracts/signal.py`
- ADRs: `../../_planning/adr/` (011, 014, 015, 016, 017, 020, 022, 023)
