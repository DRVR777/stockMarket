# MANIFEST — oracle

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle |
| `type` | project |
| `depth` | 2 |
| `parent` | programs/ |
| `version` | 0.1.0 |
| `status` | scaffold |
| `prd_source` | inline — ORACLE PRD v0.1 DRAFT |
| `created` | 2026-03-14T00:00:00Z |

## What I Am
An AI-native trading intelligence platform that monitors Polymarket and Solana markets, synthesizes multi-source signals through a multi-pass reasoning engine, executes positions, and compounds its edge over time through a self-curating post-mortem knowledge base.

## What I Contain
| Name | Type | Status | Purpose |
|------|------|--------|---------|
| CLAUDE.md | file | active | Depth-1 map: program names and one-line purposes |
| CONTEXT.md | file | active | Task router for this project |
| docker-compose.yml | file | active | Starts Redis (required before any program runs) |
| Makefile | file | active | `make up` / `make install` / `make [program]` |
| .env.example | file | active | All required env vars with descriptions — copy to .env |
| oracle-shared/ | folder | active | Pydantic contract models — installed as a package by all programs |
| _planning/ | folder | active | Architecture decisions before code |
| _meta/ | folder | active | Project-internal gap registry |
| shared/ | folder | active | Contract shapes (markdown source of truth for oracle-shared/) |
| programs/signal-ingestion | folder | scaffold | SIL — polls all data sources, normalizes to Signal objects |
| programs/whale-detector | folder | scaffold | WADE — on-chain surveillance, anomaly scoring, copy-trade surface |
| programs/osint-fusion | folder | scaffold | OSFE — semantic embedding, market association, rolling semantic state |
| programs/solana-executor | folder | scaffold | SOE — autonomous mean-reversion trader for Solana-native assets |
| programs/reasoning-engine | folder | scaffold | RE — multi-pass AI reasoning, produces TradeThesis objects |
| programs/knowledge-base | folder | scaffold | KBPM — markdown vault, post-mortem generation, self-improving loop |
| programs/operator-dashboard | folder | scaffold | Real-time operator UI — alerts, theses, PnL, parameter control |

## What I Need From Parent
Nothing — self-contained. Cross-project dependencies go in `{root}/_meta/contracts/`.

## What I Return To Parent
A continuously running trading intelligence system producing: executed trades on Polymarket and Solana, a growing post-mortem knowledge base, and an operator dashboard showing live system state.

## Routing Rules
| Condition | Go To |
|-----------|-------|
| Make an architectural decision | _planning/CONTEXT.md |
| Work on a specific program | programs/[name]/CONTEXT.md |
| Define or update a contract | shared/MANIFEST.md |
| Log a project-internal gap | _meta/gaps/pending.txt |
| Log a cross-project gap | {root}/_meta/gaps/pending.txt |
| Orient with no prior context | This MANIFEST, then CLAUDE.md |

## Gap Status
All gaps closed. 23 ADRs accepted. 10 contracts defined.
All 7 programs spec-review PASS — status: specced.
Ready to build. Start with signal-ingestion (Phase 1).
