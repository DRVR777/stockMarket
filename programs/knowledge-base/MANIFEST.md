# MANIFEST — programs/knowledge-base

## Envelope
| Field | Value |
|-------|-------|
| `id` | oracle-programs-knowledge-base |
| `type` | program |
| `depth` | 4 |
| `parent` | oracle/programs/ |
| `status` | specced |

## What I Am
The Knowledge Base & Post-Mortem System (KBPM). Maintains the markdown vault, indexes theses in ChromaDB for RE retrieval, generates post-mortems via Claude, propagates source credibility updates to OSFE.

## External Dependencies
| Depends On | What | Contract Location |
|------------|------|-------------------|
| reasoning-engine | TradeThesis objects | ../../shared/contracts/trade-thesis.md |
| signal-ingestion, solana-executor | TradeExecution objects | ../../shared/contracts/trade-execution.md |
| whale-detector | WalletProfile via AnomalyEvent | ../../shared/contracts/wallet-profile.md |
| event bus — Redis pub/sub (ADR-014) | subscribe ThesisObjects + Executions + AnomalyEvents, publish PostMortem | ../../shared/contracts/post-mortem.md |
| ChromaDB (ADR-013) | oracle_theses collection | external |
| Anthropic API (ADR-023) | claude-sonnet-4-6 for post-mortem generation | external |
| OpenAI API (ADR-012) | text-embedding-3-small for thesis indexing | external |

## Gap Status
All gaps closed. All ADRs accepted. Spec-review PASS — 2026-03-14.

## What I Produce
Markdown vault files (markets/, wallets/, signals/, theses/, osint/). PostMortem objects on `oracle:post_mortem`.

## Routing Rules
| Condition | Go To |
|-----------|-------|
| Build a feature | CONTEXT.md — follow build sequence row by row |
| Architecture question | ../../_planning/CONTEXT.md |
| Update a contract | ../../shared/contracts/ first, then return here |
