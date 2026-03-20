# ADR-005: Knowledge Base Is a File-System Markdown Vault

## Status
accepted — stated explicitly in PRD Section 2.6

## Context
PRD states: "All knowledge is stored as structured markdown documents in a hierarchical vault. Document categories include: /markets/, /wallets/, /signals/, /theses/, /osint/."

## Decision
The KBPM stores all persistent intelligence as structured markdown files in a file-system vault with the following directory structure:
- `/markets/` — one document per tracked Polymarket market
- `/wallets/` — one document per tracked wallet
- `/signals/` — indexed signal archive with semantic tags
- `/theses/` — all generated trade theses with outcome labels
- `/osint/` — curated intelligence summaries by topic domain

The vault location is a configurable root path. No database is used for document storage.

## Consequences
- Documents are human-readable without any tooling.
- RE can search the vault via semantic similarity (ADR-013) to surface analogous historical situations.
- Post-mortem generation writes back to the relevant /markets/ document.
- File naming conventions must be defined during KBPM build phase.

## Alternatives Considered
To be completed during planning phase.
