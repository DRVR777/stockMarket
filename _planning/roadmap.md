# ORACLE — Build Roadmap

## Dependency Graph

Build order is determined by contract dependencies. A program cannot build until:
1. All its assumption ADRs are accepted
2. All contracts it consumes are defined (not stub)
3. spec-review passes

```
signal-ingestion (SIL)
    ↓ produces: Signal
whale-detector (WADE) ← Signal
osint-fusion (OSFE)   ← Signal
    ↓ produces: Insight, MarketState
reasoning-engine (RE) ← Insight, MarketState, AnomalyEvent
    ↓ produces: TradeThesis
knowledge-base (KBPM) ← TradeThesis, TradeExecution → PostMortem → (feeds back to RE)
solana-executor (SOE) ← TradeThesis → TradeExecution
operator-dashboard    ← AnomalyEvent, TradeThesis, OperatorAlert (read-only consumer)
```

## Phase 0 — Resolve All Blockers (no code)

**Before writing a single line of code:**

1. Validate ADR-014 (event bus) — blocks everything. Decide: Redis pub/sub, in-process EventEmitter, Kafka, or other.
2. Validate ADR-015 (shared state) — blocks everything. Decide: Redis, PostgreSQL, SQLite, or other.
3. Validate ADR-011 (Polygon RPC) — blocks SIL and WADE.
4. Validate ADR-016 (Solana oracle) — blocks SOE and SIL.
5. Validate ADR-017 (Polymarket wallet) — blocks execution path in SIL.
6. Validate ADR-012 (embedding model) — blocks OSFE.
7. Validate ADR-013 (vector store) — blocks OSFE and RE.
8. Validate ADR-018 (dashboard delivery) — blocks operator-dashboard.
9. Validate ADR-019 (notification delivery) — blocks WADE and dashboard.
10. Validate ADR-020 (RE scheduler) — blocks RE.
11. Validate ADR-021 (SOE wallet) — blocks SOE.
12. Validate ADR-022 (OSINT sources at launch) — blocks SIL and OSFE.

After all assumption ADRs are accepted → define all contract shapes in shared/contracts/.

## Phase 1 — signal-ingestion

First program to build. It is the data source for everything else.
No other program can be tested without it producing Signal objects.

Milestone: SIL runs, polls Polymarket REST, emits Signal objects to event bus.

## Phase 2 — whale-detector + osint-fusion (parallel)

Both consume Signal. Can be built in parallel once SIL is stable.

WADE milestone: Large order detected, wallet profile looked up, AnomalyEvent emitted.
OSFE milestone: Signal embedded, matched to a market, MarketState updated.

## Phase 3 — reasoning-engine

Depends on Insight and MarketState from OSFE, AnomalyEvent from WADE.
The most complex program — build last among the core intelligence stack.

Milestone: Full context assembled for one market, adversarial hypotheses generated, TradeThesis emitted.

## Phase 4 — knowledge-base

Depends on TradeThesis and TradeExecution shapes being defined.
Post-mortem generation requires RE — build KBPM after RE is stable.

Milestone: One market document written, one post-mortem generated after resolution.

## Phase 5 — solana-executor

Can be built in parallel with knowledge-base. Shares RE but doesn't depend on KBPM.
Lowest risk to build incrementally — start with paper trading (log signals, don't execute).

Milestone: SOE detects a buy signal, logs it. Execution gated by operator.

## Phase 6 — operator-dashboard

Build last. Consumes everything but produces nothing critical.
Start with the simplest viable interface (even a CLI TUI) and upgrade later.

Milestone: Operator can see live AnomalyEvents and active TradeTheses.
