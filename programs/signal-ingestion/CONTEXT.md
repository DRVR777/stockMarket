# programs/signal-ingestion — Task Router

## What This Program Is
The sensory system of ORACLE. Polls and subscribes to every data source, normalizes output into canonical Signal objects, publishes to the shared event bus.

---

## Before Writing Any Code

**Stop 1 — ADR check:** All ADRs accepted (ADR-011, 014, 015, 016, 017, 022, 023). No stops remain.

**Stop 2 — Contract check:** signal.md is defined. No stops remain.

**Stop 3 — Spec review:** Run this after any CONTEXT.md change. OVERALL: PASS → proceed.

---

## Task Routing

Build adapters in this exact sequence. Each row is one discrete unit of work.

| Your Task | Load These | Interface |
|-----------|-----------|-----------|
| 1. Polymarket REST adapter | signal.md, ADR-014, ADR-023 | Poll `GET /markets` every 60s. Emit `Signal(source_id=polymarket_rest, category=price)` per market. |
| 2. Polymarket WebSocket adapter | signal.md, ADR-014, ADR-023 | Subscribe to WS price feed. Emit `Signal(source_id=polymarket_ws, category=price)` per tick. |
| 3. Polygon on-chain listener | signal.md, ADR-011, ADR-014 | Subscribe to Alchemy WS for `OrderFilled`/`OrderPlaced` on CLOB contract. Emit `Signal(source_id=polygon_clob, category=on_chain)` per event. |
| 4a. NewsAPI adapter | signal.md, ADR-022, ADR-014 | Poll `GET /everything` every 5min with market-derived keywords. Emit `Signal(source_id=newsapi, category=news)` per article. |
| 4b. Wikipedia adapter | signal.md, ADR-022, ADR-014 | Poll Recent Changes API every 15min. Emit `Signal(source_id=wikipedia, category=news)` per relevant edit. |
| 4c. Reddit adapter | signal.md, ADR-022, ADR-014 | Poll r/Polymarket, r/PredictIt, r/politics, r/sports every 10min via PRAW. Emit `Signal(source_id=reddit, category=social)` per post. |
| 5. Birdeye price adapter | signal.md, ADR-016, ADR-014 | Subscribe to Birdeye WS for configured Solana assets. Emit `Signal(source_id=birdeye, category=price)` per price update. |
| 6. AI opinion poller | signal.md, ADR-020, ADR-014 | Query RE for market outlook on scheduled interval. Emit `Signal(source_id=ai_opinion, category=ai_generated)` per response. |
| Fix a polling/subscription bug | The failing adapter file, signal.md | Reproduce → fix → verify Signal output matches signal.md shape. |
| Update the Signal shape | signal.md, _planning/adr/ | Update contract first. Update all adapter normalization. Do not touch src/ until contract is updated. |

---

## Shared Adapter Pattern

Every adapter follows the same structure:
```
class [Source]Adapter:
    async def start(self) -> None          # connect / subscribe
    async def stop(self) -> None           # disconnect cleanly
    async def _normalize(self, raw) -> Signal  # raw → Signal shape
    async def _publish(self, signal: Signal)   # redis.publish("oracle:signal", signal.model_dump_json())
```

All adapters use `redis.asyncio` client shared from a single connection pool at SIL startup.

---

## Checkpoints

| After step | Present | Human options |
|------------|---------|---------------|
| Adapter 1 complete | Sample Signal JSON from live Polymarket REST call | approve / revise shape |
| Adapter 3 complete | Sample on-chain Signal from live Polygon event | approve / revise |
| All adapters complete | Signal count per category over 1 hour of live data | approve for Audit / revise |

---

## Inference Logging
All inferences → `../../_meta/gaps/pending.txt`
Format: `[timestamp] | signal-ingestion | [file] | inferred "[what]" — no file states this`
