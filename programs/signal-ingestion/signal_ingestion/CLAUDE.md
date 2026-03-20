# CLAUDE.md — signal_ingestion

## What Is In This Directory

| Name | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `__main__.py` | Async entry point — starts all 8 adapters concurrently |
| `config.py` | Centralized env-var-backed configuration constants |
| `adapters/` | One module per data source (8 adapters total) |

## Quick Rules For This Directory

- Import contracts from `oracle_shared.contracts`, never redefine
- Import config from `signal_ingestion.config`, never use raw `os.getenv` in adapters
- All adapters implement: `start()`, `stop()`, `_normalize()`, `_publish()`
- Channel names come from contract classes (`Signal.CHANNEL`)

## Cross-References

- `oracle_shared.contracts.signal` — Signal, SignalCategory, SourceId
