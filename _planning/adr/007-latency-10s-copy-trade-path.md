# ADR-007: On-Chain Fill to Copy-Trade Decision Latency Under 10 Seconds

## Status
accepted — stated explicitly in PRD Section 4

## Context
PRD states: "The path from an on-chain order fill to a copy-trade execution decision must complete in under 10 seconds under normal conditions."

## Decision
The critical latency path — on-chain OrderFilled event → WADE anomaly scoring → operator alert / auto-execute — must complete end-to-end in under 10 seconds under normal load.

This latency budget applies only to the copy-trade path through SIL → WADE → executor. The RE's full market scan is explicitly asynchronous and is not subject to this constraint.

## Consequences
- SIL must subscribe to on-chain events via WebSocket (not polling) for the WADE path.
- WADE anomaly scoring must be O(1) against the wallet registry — no full-table scans.
- The copy-trade execution path must not block on RE reasoning.
- Performance testing of the SIL → WADE → executor path is required before live trading.

## Alternatives Considered
To be completed during planning phase.
