# oracle — Project Map

## Hard Rule
Depth 1 only. Program names and one-line purposes.
Internal structure of each program lives in that program's own CONTEXT.md.
If program internals appear here, remove them.

---

## Programs

| Program | Purpose | Status |
|---------|---------|--------|
| `signal-ingestion` | SIL — continuously polls Polymarket REST/WS, Polygon on-chain events, OSINT feeds, Solana oracles, and AI opinion streams; normalizes everything into canonical Signal objects | active |
| `whale-detector` | WADE — watches on-chain fills for large orders, scores anomalies, maintains wallet registry, surfaces copy-trade opportunities | active |
| `osint-fusion` | OSFE — embeds incoming signals, runs semantic similarity against active markets, maintains rolling semantic state per market | scaffold |
| `solana-executor` | SOE — autonomous mean-reversion trader: monitors configured assets, computes statistical model + AI floor estimate, executes entries and exits | scaffold |
| `reasoning-engine` | RE — multi-pass reasoning: assembles context → generates adversarial hypotheses → weights evidence → calibrates confidence → emits TradeThesis | scaffold |
| `knowledge-base` | KBPM — writes and maintains the markdown vault (/markets/, /wallets/, /signals/, /theses/, /osint/); generates post-mortems; feeds RE via semantic search | scaffold |
| `operator-dashboard` | Real-time operator interface: live anomaly alerts, active theses with confidence, SOE positions with PnL, post-mortem feed, parameter control panel | scaffold |

---

## Workspace Rules

1. An agent in one program never loads another program's src/.
2. All cross-program data shapes live in shared/contracts/.
3. Check _planning/adr/ before writing code — 22 ADRs exist; 12 are assumptions that block building.
4. Log every inference to _meta/gaps/pending.txt during a task, not after.
5. Fix-first: when an error or broken reference is found, fix it without asking.

---

## Navigation

| You want to... | Go to |
|----------------|-------|
| Plan or decide architecture | _planning/CONTEXT.md |
| Work on a specific program | programs/[name]/CONTEXT.md |
| Find a contract | shared/contracts/ |
| See open gaps | _meta/gaps/CONTEXT.md |
| See all architectural decisions | _planning/adr/ |
| Run spec review on a program | {root}/_meta/spec-review.md |
| Look up an architectural pattern | {root}/_core/CONVENTIONS.md |

---

## What to Load

| Task | Load these files | Do NOT load |
|------|-----------------|-------------|
| Start a new program | MANIFEST.md, _planning/CONTEXT.md, _planning/prd-source.md | other programs' files |
| Work on an existing program | programs/[name]/CONTEXT.md, programs/[name]/MANIFEST.md, shared/contracts/ | _planning/prd-source.md, other programs |
| Review architecture decision | _planning/adr/[relevant ADR] | program source files |
| Run spec review | programs/[name]/CONTEXT.md, programs/[name]/MANIFEST.md | output/ folders |
| Close a gap | _meta/gaps/CONTEXT.md, the gap JSON file | unrelated program files |
| Find a contract | shared/MANIFEST.md, shared/contracts/[name].md | program source files |
