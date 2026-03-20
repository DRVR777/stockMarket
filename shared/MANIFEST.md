# MANIFEST — oracle/shared

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-shared |
| `type` | contracts |
| `depth` | 3 |
| `parent` | oracle/ |
| `status` | active |

## What I Am
The hard boundary between programs. Programs never import from each other.
If two programs need to agree on a data shape — the definition lives here.
All contract Pydantic models are also defined in the `oracle-shared` Python package (see ADR-023).

## The Rule
If it's not defined here, both sides will infer independently → divergence → missing_bridge gap.

## Contracts

| Contract | Produced By | Consumed By | Status | Redis Channel |
|----------|-------------|-------------|--------|---------------|
| signal.md | signal-ingestion | whale-detector, osint-fusion | **defined** | `oracle:signal` |
| anomaly-event.md | whale-detector | reasoning-engine, operator-dashboard | **defined** | `oracle:anomaly_event` |
| insight.md | osint-fusion | reasoning-engine | **defined** | `oracle:insight` |
| market-state.md | osint-fusion | reasoning-engine | **defined** | `oracle:market_state` |
| trade-thesis.md | reasoning-engine | knowledge-base, solana-executor, operator-dashboard | **defined** | `oracle:trade_thesis` |
| wallet-profile.md | whale-detector | knowledge-base | **defined** | Redis state only (no channel) |
| trade-execution.md | signal-ingestion, solana-executor | knowledge-base | **defined** | `oracle:trade_execution` |
| post-mortem.md | knowledge-base | osint-fusion, operator-dashboard | **defined** | `oracle:post_mortem` |
| operator-alert.md | whale-detector, reasoning-engine | operator-dashboard | **defined** | `oracle:operator_alert` |
| copy-trade-approval.md | operator-dashboard | signal-ingestion | **defined** | `oracle:copy_trade_approved` |

## oracle-shared Python package
Pydantic model classes for all contracts live in `oracle-shared/` package.
Install in each program: `uv pip install -e ../../oracle-shared`
Import: `from oracle_shared.contracts import Signal, TradeThesis, ...`
