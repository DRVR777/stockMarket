# MANIFEST — whale_detector

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-whale-detector-pkg |
| `type` | package |
| `depth` | 5 |
| `parent` | whale-detector/ |
| `status` | active |

## What I Am
Python package implementing WADE — signal subscriber, anomaly scorer, wallet registry, cascade detector, and event emitter.

## Contents
| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `__main__.py` | Async entry point — subscribes to signals and runs the detection pipeline |
| `config.py` | Centralized env-var-backed configuration constants |
| `signal_subscriber.py` | Step 1: Subscribe to `oracle:signal`, filter on-chain polygon_clob signals |
| `threshold_flagger.py` | Step 2: Flag large orders exceeding configurable USD threshold |
| `wallet_registry.py` | Step 3 & 8: Lookup/create/update WalletProfile in Redis |
| `anomaly_scorer.py` | Step 4: Compute weighted anomaly score from size, wallet history, timing |
| `cascade_detector.py` | Step 5: Detect coordinated multi-wallet activity within time windows |
| `event_emitter.py` | Steps 6 & 7: Assemble and publish AnomalyEvent and OperatorAlert |
