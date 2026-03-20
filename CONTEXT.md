# oracle — Task Router

## What This Project Is
An AI-native trading intelligence platform that monitors Polymarket and Solana markets, synthesizes multi-source signals through a multi-pass reasoning engine, executes positions, and compounds its edge through a self-curating post-mortem knowledge base.

---

## Task Routing

| Your Task | Go To | Also Load |
|-----------|-------|-----------|
| Plan architecture or make a technical decision | _planning/CONTEXT.md | Nothing else yet |
| Work on signal-ingestion (SIL) | programs/signal-ingestion/CONTEXT.md | shared/contracts/signal.md |
| Work on whale-detector (WADE) | programs/whale-detector/CONTEXT.md | shared/contracts/signal.md, shared/contracts/anomaly-event.md, shared/contracts/wallet-profile.md, shared/contracts/operator-alert.md |
| Work on osint-fusion (OSFE) | programs/osint-fusion/CONTEXT.md | shared/contracts/signal.md, shared/contracts/insight.md, shared/contracts/market-state.md |
| Work on solana-executor (SOE) | programs/solana-executor/CONTEXT.md | shared/contracts/trade-thesis.md, shared/contracts/trade-execution.md |
| Work on reasoning-engine (RE) | programs/reasoning-engine/CONTEXT.md | shared/contracts/insight.md, shared/contracts/anomaly-event.md, shared/contracts/market-state.md, shared/contracts/trade-thesis.md |
| Work on knowledge-base (KBPM) | programs/knowledge-base/CONTEXT.md | shared/contracts/trade-thesis.md, shared/contracts/trade-execution.md, shared/contracts/post-mortem.md, shared/contracts/wallet-profile.md |
| Work on operator-dashboard | programs/operator-dashboard/CONTEXT.md | shared/contracts/anomaly-event.md, shared/contracts/trade-thesis.md, shared/contracts/operator-alert.md |
| Define or update a contract | shared/MANIFEST.md | _planning/adr/ for context |
| Log an inference or gap | _meta/gaps/pending.txt | Nothing |

---

## Status
Scaffolded from PRD. Programs are stubs. 12 assumption ADRs must be validated before building.
Work in _planning/ before writing any code in programs/.
Build order defined in _planning/roadmap.md.
