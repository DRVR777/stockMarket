# MANIFEST — programs/operator-dashboard

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-programs-operator-dashboard |
| `type` | program |
| `depth` | 4 |
| `parent` | oracle/programs/ |
| `status` | specced |

## What I Am
Real-time operator interface. FastAPI + vanilla JS at localhost:8080. Subscribes to all Oracle event bus channels, pushes live data to the browser via WebSocket, handles copy-trade approval and parameter changes. Relays action-required alerts to Telegram.

## External Dependencies
| Depends On | What | Contract Location |
|------------|------|-------------------|
| all programs | all event types via event bus | all contracts in ../../shared/contracts/ |
| event bus — Redis pub/sub (ADR-014) | subscribe all oracle:* channels | internal |
| shared state — Redis (ADR-015) | read/write oracle:state:params | internal |
| Telegram bot (ADR-019) | push action_required alerts | external |
| FastAPI + uvicorn (ADR-018) | serve dashboard at localhost:8080 | external |

## Gap Status
All gaps closed. All ADRs accepted. Spec-review PASS — 2026-03-14.

## What I Produce
Operator actions written to Redis and published to event bus. Telegram notifications on action_required alerts.

## Routing Rules
| Condition | Go To |
|-----------|-------|
| Build a feature | CONTEXT.md — follow build sequence row by row |
| Architecture question | ../../_planning/CONTEXT.md |
| Update a contract | ../../shared/contracts/ first, then return here |
