# ADR-006: Capital Safety Circuit Breakers at Infrastructure Level

## Status
accepted — stated explicitly in PRD Section 4

## Context
PRD states: "Hard-coded circuit breakers at the infrastructure level. Maximum daily loss limits must be enforced below the application layer where possible."

## Decision
Capital safety constraints are enforced at two levels:
1. **Infrastructure level** — maximum daily loss limits enforced by the execution infrastructure, not application code. If application code crashes or misbehaves, capital exposure is still bounded.
2. **Application level** — each executor (Polymarket and SOE) maintains its own circuit breaker: max concurrent positions, max capital per position, daily loss ceiling. If the daily loss ceiling is hit, all execution pauses and the operator is alerted.

SOE-specific circuit breaker (from PRD Section 2.4): if cumulative daily loss exceeds a configurable threshold, all SOE activity pauses.

## Consequences
- Execution programs must implement circuit breaker logic before any live trading.
- Infrastructure-level enforcement mechanism is deployment-specific (resolved at build time).
- Circuit breaker thresholds are operator-configurable via the operator-dashboard.
- Paper trading mode (logging signals without executing) bypasses circuit breakers but mirrors the same decision path.

## Alternatives Considered
To be completed during planning phase.
