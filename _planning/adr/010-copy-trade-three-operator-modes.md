# ADR-010: Copy-Trade Has Three Operator Modes

## Status
accepted — stated explicitly in PRD Section 3

## Context
PRD states: "Copy-trade approval can be configured in three modes: fully manual (operator approves each), semi-automatic (auto-execute below a size threshold, prompt above), or fully automatic (execute all signals above the anomaly score threshold)."

## Decision
The copy-trade execution path supports three operator-configurable modes:
1. **Fully manual** — every WADE-flagged copy-trade requires explicit operator approval before execution
2. **Semi-automatic** — auto-execute if position size ≤ configured threshold AND anomaly score ≥ configured minimum; prompt operator for larger positions
3. **Fully automatic** — execute all signals where anomaly score ≥ configured threshold, up to the wallet's configured position scale

The active mode is a runtime parameter configurable via the operator-dashboard without system restart.

## Consequences
- The copy-trade execution path must check mode before executing.
- Modes 2 and 3 require circuit breakers (ADR-006) to be active before enabling.
- Mode transitions are logged to KBPM as operator actions.
- Default mode at startup: fully manual (safest default).

## Alternatives Considered
To be completed during planning phase.
