# ADR-015: Shared State Layer

## Status
accepted

## Context
Programs need to read current runtime state without replaying the event stream: active market list, wallet registry, open positions, circuit breaker flags, operator parameters.

## Decision
**Redis** — same Docker instance as ADR-014, separate key namespace prefix `oracle:state:`.

Key schema:
- `oracle:state:markets` — Hash: market_id → MarketState JSON (updated by OSFE)
- `oracle:state:wallets` — Hash: wallet_address → WalletProfile JSON (updated by WADE)
- `oracle:state:positions` — Hash: execution_id → TradeExecution JSON (updated by executors)
- `oracle:state:circuit_breaker` — Hash: program_name → {"active": bool, "triggered_at": ISO-8601}
- `oracle:state:params` — Hash: param_name → value (operator-configurable thresholds)
- `oracle:state:daily_pnl` — String: float, reset at midnight UTC

All values are JSON strings. TTLs: markets expire 48h after resolution, positions expire 90 days after close.

Python client: `redis.asyncio` — same client instance used for pub/sub (different connection pool).

## Consequences
- WADE reads/writes `oracle:state:wallets` — wallet registry lives in Redis, not a database
- Operator parameter changes via dashboard write to `oracle:state:params` — programs read on each decision cycle
- If Redis restarts, state is lost unless Redis persistence (AOF) is enabled — enable AOF in Docker config
- For production resilience: enable Redis AOF persistence (`appendonly yes`)

## Alternatives Considered
- PostgreSQL: better for complex queries, more ops overhead than needed at this scale
- SQLite: single-writer bottleneck under concurrent program access
- Separate Redis + Postgres: most robust, more complex — revisit if state size grows beyond Redis comfort zone
