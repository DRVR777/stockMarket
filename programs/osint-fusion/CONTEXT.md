# programs/osint-fusion — Task Router

## What This Program Is
OSFE — interpretive cortex. Subscribes to `oracle:signal`, embeds signal text via OpenAI, runs ChromaDB similarity search against active market registry, maintains rolling semantic state per market, emits Insight and MarketState objects.

---

## Before Writing Any Code

All ADRs accepted. All contracts defined. No stops remain — proceed after spec-review PASS.

---

## Task Routing

Build in this exact sequence. Each row is one discrete unit of work.

| Your Task | Load These | Specification |
|-----------|-----------|---------------|
| 1. ChromaDB setup | ADR-013, market-state.md | Initialize `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)`. Create collection `oracle_markets` if not exists (embedding dimension 512, cosine distance). On startup: populate from `oracle:state:markets` Redis hash — embed each `market_question` and upsert into collection with `id=market_id`. |
| 2. Signal subscriber | signal.md, ADR-014 | Subscribe to `oracle:signal`. For each message: extract `raw_text` (see extraction rules below). If `raw_text` is empty or < 10 chars → discard. Pass to embedding step. |
| 3. Embedding pipeline | insight.md, ADR-012 | Batch signals: accumulate for up to 500ms, then call `openai.embeddings.create(model="text-embedding-3-small", input=[...texts], dimensions=512)`. One embedding per signal. |
| 4. Similarity search | insight.md, ADR-013 | For each embedding: `collection.query(query_embeddings=[vec], n_results=10)`. Keep only results where distance ≤ 0.35 (= cosine similarity ≥ 0.65, per threshold in insight.md). Build `similarity_scores = {market_id: 1-distance}`. |
| 5. Credibility weighting | insight.md | Apply `source_credibility_weight` from the defaults table in insight.md (keyed by Signal.category). Weights are stored in `oracle:state:params:credibility_weights:{category}` and override defaults when present. |
| 6. Insight emit | insight.md, ADR-014 | Assemble Insight object. `raw_text` = extracted text from step 2. `semantic_summary` = first 500 chars of raw_text (full summarization deferred to RE). Publish to `oracle:insight`. |
| 7. MarketState update | market-state.md, ADR-015 | For each market_id in similarity_scores: read current MarketState from `oracle:state:markets:{market_id}`. Prepend new Insight to `recent_insights` list, trim to 20. Increment `signal_count_24h`. Regenerate `semantic_state_summary` via Claude (`claude-haiku-4-5-20251001`, summarize last 5 insights in ≤ 100 words — Haiku for cost). Write back to Redis. Publish updated MarketState to `oracle:market_state`. |
| 8. PostMortem weight updater | post-mortem.md, ADR-015 | Subscribe to `oracle:post_mortem`. For each PostMortem: apply `source_weight_updates` deltas to `oracle:state:params:credibility_weights:{source_id}`. Clamp result to 0.1–2.0. |
| Fix a bug | The failing unit, the relevant contract | Reproduce → fix → verify output matches contract shape. |

---

## raw_text extraction rules by source_id

| source_id | raw_text field |
|-----------|---------------|
| newsapi | `title + ". " + description` |
| wikipedia | `page_title + ": " + summary` |
| reddit | `title` |
| polymarket_rest | `question` |
| polygon_clob | skipped — no text content, handled by WADE not OSFE |
| birdeye | skipped — price data, not text |
| ai_opinion | `response_text` |

---

## Checkpoints

| After step | Present | Human options |
|------------|---------|---------------|
| Step 4 complete | Top 3 market matches for 5 sample signals with scores | approve threshold / adjust |
| Step 7 complete | Sample MarketState JSON after 1 hour of live signals | approve / revise summary quality |
| All steps complete | Insight volume/hour, avg markets matched per signal | approve for Audit / revise |

---

## Inference Logging
All inferences → `../../_meta/gaps/pending.txt`
Format: `[timestamp] | osint-fusion | [file] | inferred "[what]" — no file states this`
