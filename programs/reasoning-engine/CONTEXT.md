# programs/reasoning-engine — Task Router

## What This Program Is
RE — intelligence core. Runs a four-step adversarial reasoning pipeline on demand (signal-triggered) and on schedule (every N minutes). Emits TradeThesis objects. Serves SOE floor estimate requests on-demand.

---

## Before Writing Any Code

All ADRs accepted. All contracts defined. No stops remain — proceed after spec-review PASS.

---

## Task Routing

Build in this exact sequence. Each row is one discrete unit of work.

| Your Task | Load These | Specification |
|-----------|-----------|---------------|
| 1. ChromaDB setup | ADR-013, trade-thesis.md | Initialize `chromadb.PersistentClient`. Create collection `oracle_theses` if not exists (dimension 512, cosine). On startup index any existing thesis documents from `oracle:state:theses_index` (Redis set of thesis_ids). |
| 2. Insight subscriber | insight.md, market-state.md, ADR-014 | Subscribe to `oracle:insight`. On each Insight: for each associated_market_id, enqueue a trigger analysis for that market. Deduplicate: if a market is already queued, skip (use Redis set `oracle:state:re_queue`). |
| 3. Step 1 — Context assembly | market-state.md, anomaly-event.md, trade-thesis.md, ADR-013 | For a given market_id: (a) read MarketState from `oracle:state:markets:{market_id}`, (b) read last 5 AnomalyEvents for this market from `oracle:state:anomaly_index:{market_id}` Redis list, (c) query `oracle_theses` ChromaDB collection with market_question embedding — retrieve top 5 analogues with similarity > 0.7. Assemble into ContextAssembly object. |
| 4. Step 2 — Hypothesis generation | trade-thesis.md, ADR-004, ADR-023 | Call Claude claude-sonnet-4-6. System prompt: "You are an adversarial prediction market analyst. Argue both sides." User prompt: ContextAssembly JSON. Request structured output: `{hypotheses: [{side, argument, evidence: [str]}]}` — exactly one YES hypothesis and one NO hypothesis. Parse response into `list[Hypothesis]`. |
| 5. Step 3 — Evidence weighting | trade-thesis.md | For each hypothesis: score against ContextAssembly.market_state.recent_insights. Score formula: `(count of recent_insights supporting this side) / len(recent_insights)` normalized 0–1, weighted by each insight's source_credibility_weight. Compute `re_probability_estimate = YES_score / (YES_score + NO_score)`. Compute `probability_delta = re_probability_estimate - market_state.current_price_yes`. If `abs(probability_delta) < oracle:state:params:re_delta_threshold` (default 0.10) → set decision=skip, emit thesis, stop. |
| 6. Step 4 — Confidence calibration | trade-thesis.md, ADR-015 | Three sub-scores averaged: (a) recency: fraction of recent_insights from last 6h vs total window, (b) diversity: unique source categories / 5, (c) model certainty: 1 - (entropy of YES_score vs NO_score, normalized). If `confidence < oracle:state:params:re_confidence_min` (default 0.45) → decision=flag_for_review. Else → decision=execute. Set `recommended_position_usd` per formula in trade-thesis.md. |
| 7. TradeThesis emit | trade-thesis.md, ADR-014 | Assemble full TradeThesis with all prior step outputs. Publish to `oracle:trade_thesis`. Store thesis_id in `oracle:state:theses_index` Redis set. |
| 8. Scheduled full scan | ADR-020, market-state.md | APScheduler job fires every N minutes (default 30, read from `oracle:state:params:re_scan_interval_minutes`). For each market_id in `oracle:state:markets` Redis hash: run Steps 3–7. This catches stale mispricings not triggered by new signals. |
| 9. SOE floor estimate handler | ADR-009, ADR-023 | Expose a Redis pub/sub request/reply pattern: subscribe to `oracle:re_floor_request`. On each message (contains `{asset, price_history: [...], request_id}`): call Claude claude-sonnet-4-6 with price history and ask for probabilistic 24–72h floor estimate. Publish response to `oracle:re_floor_response:{request_id}`. |
| 10. AnomalyEvent index | anomaly-event.md, ADR-015 | Subscribe to `oracle:anomaly_event`. On each event: LPUSH to `oracle:state:anomaly_index:{market_id}`, LTRIM to last 10. Used by Step 3. |
| Fix a bug | The failing step file, relevant contracts | Reproduce → fix → verify TradeThesis output matches contract shape. |

---

## Checkpoints

| After step | Present | Human options |
|------------|---------|---------------|
| Step 5 complete | 3 sample probability estimates vs actual market prices | approve delta formula / adjust threshold |
| Step 7 complete | 1 full TradeThesis JSON from live data | approve / revise reasoning quality |
| Step 8 complete | Scan time for all active markets, thesis rate per hour | approve for Audit / revise |

---

## Inference Logging
All inferences → `../../_meta/gaps/pending.txt`
Format: `[timestamp] | reasoning-engine | [file] | inferred "[what]" — no file states this`
