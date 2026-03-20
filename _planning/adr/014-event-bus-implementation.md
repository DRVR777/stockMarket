# ADR-014: Event Bus Implementation

## Status
accepted

## Context
All 7 programs communicate via a shared event bus. Must support multiple concurrent publishers and subscribers, survive individual program restarts, and deliver messages with low latency (the SIL→WADE→executor path must be under 10 seconds, ADR-007).

## Decision
**Redis pub/sub** — deployed via Docker (`redis:7-alpine`), port 6379.
- Each program connects independently: if one crashes, the bus keeps running
- Pub/sub channels map to contract types: `oracle:signal`, `oracle:anomaly_event`, `oracle:insight`, `oracle:market_state`, `oracle:trade_thesis`, `oracle:trade_execution`, `oracle:post_mortem`, `oracle:operator_alert`
- Python client: `redis-py` with async support (`redis.asyncio`)
- Message format: JSON-serialized contract objects

**Channel naming convention:** `oracle:[contract_slug]` — e.g. `oracle:signal`, `oracle:trade_thesis`

Redis instance is shared with ADR-015 (shared state) via separate key namespaces. Same Docker container, different key prefixes.

## Consequences
- Every program must handle reconnect logic if Redis is temporarily unavailable
- Messages are not durable (pub/sub is fire-and-forget) — programs that miss a message while down will not receive it. Acceptable: SIL continuously re-polls sources; KBPM catches up via periodic vault sync
- If durability becomes critical post-launch, migrate the `oracle:trade_thesis` channel to Redis Streams

## Alternatives Considered
- Redis Streams: durable, consumer groups, more complex — worthwhile upgrade if fire-and-forget causes issues
- In-process asyncio Queue: forces all programs into one process — incompatible with modularity requirement
- Kafka: heavy ops overhead, overkill at this scale
