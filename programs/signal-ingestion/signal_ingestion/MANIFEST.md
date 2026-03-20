# MANIFEST — signal_ingestion

## Envelope

| Field       | Value                    |
|-------------|--------------------------|
| name        | signal_ingestion                   |
| type        | program-submodule                  |
| depth       | 5                  |
| status      | active                 |
| path        | programs/oracle/programs/signal-ingestion/signal_ingestion                    |

## Purpose

Python package for the Signal Ingestion Layer (SIL). Contains the entry point (`__main__.py`),
centralized configuration (`config.py`), and the `adapters/` sub-package with one module per
data source.

## Contents

  - `__init__.py` — package init with module docstring
  - `__main__.py` — async entry point, starts all adapters concurrently
  - `config.py` — all env-var-backed configuration constants (no magic numbers)
  - `adapters/` — one adapter module per data source (8 total)

## Needs

- `oracle_shared.contracts.signal` — Signal, SignalCategory, SourceId models
- Redis server at `REDIS_URL`
- External API keys per adapter (see `.env.example`)

## Returns

- Signal objects published to `oracle:signal` Redis channel

## Gap Status

- open: none
- closed: none
