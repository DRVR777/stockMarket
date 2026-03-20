# programs/knowledge-base — Task Router

## What This Program Is
KBPM — memory and learning layer. Writes the markdown vault, indexes theses for RE semantic search, generates post-mortems on close, propagates source weight updates to OSFE.

---

## Before Writing Any Code

All ADRs accepted. All contracts defined. No stops remain — proceed after spec-review PASS.

---

## Vault Structure (from ADR-005)

```
vault/                              (root = KBPM_VAULT_DIR env var)
├── markets/   {market_id}.md       one file per Polymarket market
├── wallets/   {wallet_address}.md  one file per tracked wallet
├── signals/   {YYYY-MM-DD}/{signal_id}.md
├── theses/    {thesis_id}.md       all TradeThesis, never deleted
└── osint/     {topic-slug}.md      curated intel summaries
```

---

## Task Routing

Build in this exact sequence. Each row is one discrete unit of work.

| Your Task | Load These | Specification |
|-----------|-----------|---------------|
| 1. Vault initializer | ADR-005, ADR-023 | On startup: create vault directory tree if not present (`os.makedirs` for all 5 subdirs). Log vault root path. No file writes at init. |
| 2. ThesisWriter — subscribe | trade-thesis.md, ADR-014 | Subscribe to `oracle:trade_thesis`. On each ThesisObject: write `vault/theses/{thesis_id}.md` with YAML front-matter (`thesis_id`, `market_id`, `decision`, `confidence_score`, `outcome: null`) followed by JSON block of full thesis. NEVER overwrite — if file exists, log warning and skip. |
| 3. ThesisIndexer — ChromaDB | trade-thesis.md, ADR-013 | After writing each thesis file: embed `market_question + " " + direction + " " + hypotheses[0].argument` via OpenAI text-embedding-3-small. Upsert into `oracle_theses` ChromaDB collection with `id=thesis_id`, metadata `{market_id, decision, confidence_score, outcome: null}`. |
| 4. MarketWriter | trade-thesis.md, ADR-015 | On each TradeThesis (decision != skip): create or append `vault/markets/{market_id}.md`. Front-matter: `market_id`, `question`, `resolution_deadline`, `status: open`. Body sections: `## Signals` (populated later), `## Theses` (append thesis_id + summary on each new thesis), `## Resolution` (empty until closed), `## Post-Mortem` (empty until generated). |
| 5. WalletWriter | wallet-profile.md, ADR-015 | Subscribe to `oracle:anomaly_event`. On each AnomalyEvent with attached wallet_profile: write/overwrite `vault/wallets/{wallet_address}.md`. Content: full WalletProfile JSON + trade history list (append each event). |
| 6. TradeExecution subscriber | trade-execution.md, ADR-014 | Subscribe to `oracle:trade_execution`. On status=open: update `vault/markets/{market_id}.md` `## Theses` section with execution_id. On status=closed: trigger post-mortem pipeline (step 7). |
| 7. Post-mortem pipeline | post-mortem.md, trade-thesis.md, trade-execution.md, ADR-023 | Triggered by: TradeExecution status=closed OR market_resolved signal (from SIL Polymarket REST polling `end_date` passed). Steps: (a) read full `vault/markets/{market_id}.md`, (b) read linked thesis from `vault/theses/{thesis_id}.md`, (c) call Claude claude-sonnet-4-6 with system "You are a trading post-mortem analyst." and the full market doc, (d) parse response into PostMortem object, (e) append `## Post-Mortem` section to market doc, (f) update `vault/theses/{thesis_id}.md` YAML front-matter `outcome` field, (g) update ChromaDB `oracle_theses` metadata for this thesis_id with `outcome`, (h) publish PostMortem to `oracle:post_mortem`. |
| 8. Outcome labeler | trade-thesis.md, ADR-015 | Subscribe to `oracle:post_mortem`. Update `oracle:state:theses_index` Redis set — no-op (thesis_id already present). Emit nothing — post-mortem already published in step 7. |
| Fix a bug | The failing unit, relevant contracts | Reproduce → fix → verify vault file output matches expected structure. |

---

## Checkpoints

| After step | Present | Human options |
|------------|---------|---------------|
| Step 3 complete | 3 sample vault/theses/*.md files + ChromaDB entry count | approve format / revise |
| Step 7 complete | 1 full post-mortem from a resolved market | approve reasoning quality / revise prompt |
| All steps complete | Vault file count, ChromaDB entry count | approve for Audit / revise |

---

## Inference Logging
All inferences → `../../_meta/gaps/pending.txt`
Format: `[timestamp] | knowledge-base | [file] | inferred "[what]" — no file states this`
