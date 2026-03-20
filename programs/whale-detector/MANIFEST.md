# MANIFEST — programs/whale-detector

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-programs-whale-detector |
| `type` | program |
| `depth` | 4 |
| `parent` | oracle/programs/ |
| `status` | active |

## What I Am
The Whale & Anomaly Detection Engine (WADE). Subscribes to on-chain Signals from SIL, scores anomalies, maintains the wallet registry in Redis, surfaces copy-trade opportunities as OperatorAlerts.

## External Dependencies
| Depends On | What | Contract Location |
|------------|------|-------------------|
| signal-ingestion | Signal objects (category: on_chain) | ../../shared/contracts/signal.md |
| event bus — Redis pub/sub (ADR-014) | subscribe Signals, publish AnomalyEvents + OperatorAlerts | ../../shared/contracts/anomaly-event.md, operator-alert.md |
| shared state — Redis (ADR-015) | wallet registry, cascade sets, params | ../../shared/contracts/wallet-profile.md |
| Telegram bot (ADR-019) | relayed via operator-dashboard | ../../shared/contracts/operator-alert.md |

## Gap Status
All gaps closed. All ADRs accepted. Spec-review PASS — 2026-03-14.

## What I Produce
AnomalyEvent objects on `oracle:anomaly_event`. OperatorAlert objects on `oracle:operator_alert`. WalletProfile updates in Redis `oracle:state:wallets`.

## Routing Rules
| Condition | Go To |
|-----------|-------|
| Build a feature | CONTEXT.md — follow build sequence row by row |
| Architecture question | ../../_planning/CONTEXT.md |
| Update a contract | ../../shared/contracts/ first, then return here |
