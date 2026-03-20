# ADR-003: OSFE Uses Vector Embeddings for Semantic Market Association

## Status
accepted — stated explicitly in PRD Section 2.3

## Context
PRD states: "Every incoming signal passes through a vectorized embedding step. The embedded signal is compared against the current active market registry using semantic similarity search... This association is probabilistic, not keyword-based."

## Decision
The OSINT Semantic Fusion Engine (OSFE) embeds every incoming Signal using a vector embedding model and performs semantic similarity search against the active Polymarket market registry to associate signals with relevant markets. Keyword matching is explicitly not used.

## Consequences
- OSFE requires an embedding model (resolved in ADR-012).
- OSFE requires a vector store for similarity search (resolved in ADR-013).
- The active market registry must be kept current — SIL is responsible for feeding it.
- Association results are probabilistic scores, not binary matches.

## Alternatives Considered
To be completed during planning phase.
