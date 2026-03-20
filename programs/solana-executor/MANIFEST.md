# MANIFEST — programs/solana-executor

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-programs-solana-executor |
| `type` | program |
| `depth` | 4 |
| `parent` | oracle/programs/ |
| `status` | specced |

## What I Am
The Solana Opportunistic Executor (SOE). Monitors Solana asset prices via Birdeye WS, maintains a 20-day statistical model per asset, queries RE for AI floor estimates via Redis request/reply, executes mean-reversion trades via Jupiter. Starts in paper trading mode — live execution requires explicit checkpoint approval.

## External Dependencies
| Depends On | What | Contract Location |
|------------|------|-------------------|
| Birdeye API (ADR-016) | real-time price ticks + OHLCV history | external |
| Jupiter v6 API (ADR-016) | swap execution routing | external |
| reasoning-engine | AI floor estimate via oracle:re_floor_request/response | ../../shared/contracts/trade-thesis.md (RE channel) |
| event bus — Redis pub/sub (ADR-014) | publish TradeExecution + OperatorAlert | ../../shared/contracts/trade-execution.md |
| shared state — Redis (ADR-015) | statistical model, circuit breaker, daily PnL | internal |
| Solana wallet (ADR-021) | SOLANA_PRIVATE_KEY env var, solders.Keypair | external |

## Gap Status
All gaps closed. All ADRs accepted. Spec-review PASS — 2026-03-14.

## What I Produce
TradeExecution objects on `oracle:trade_execution`. OperatorAlert on circuit breaker events.

## Routing Rules
| Condition | Go To |
|-----------|-------|
| Build a feature | CONTEXT.md — follow build sequence row by row |
| Architecture question | ../../_planning/CONTEXT.md |
| Update a contract | ../../shared/contracts/ first, then return here |
