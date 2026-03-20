# MANIFEST — tests

## Envelope

| Field       | Value                    |
|-------------|--------------------------|
| name        | tests                   |
| type        | test-suite               |
| depth       | 5                  |
| status      | active                 |
| path        | programs/oracle/programs/signal-ingestion/tests                    |

## Purpose

Live integration tests for each signal-ingestion adapter. Each test uses a FakeRedis
in-memory stub (no server required) and validates that published Signals conform to
the canonical schema in `oracle_shared.contracts.signal`.

## Contents

  - `test_polymarket_rest.py` — Adapter 1: Polymarket REST
  - `test_polymarket_ws.py` — Adapter 2: Polymarket WebSocket
  - `test_newsapi.py` — Adapter 4a: NewsAPI
  - `test_wikipedia.py` — Adapter 4b: Wikipedia
  - `test_reddit.py` — Adapter 4c: Reddit
  - `test_birdeye.py` — Adapter 5: Birdeye

## Needs

- Network access to external APIs for live tests
- `oracle_shared` and `signal_ingestion` packages installed

## Returns

- Pass/fail results per adapter
