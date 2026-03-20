# ADR-012: Embedding Model for OSFE Semantic Search

## Status
accepted

## Context
OSFE must embed every incoming Signal and every active Polymarket market description for semantic similarity matching. Also needed by KBPM to index /theses/ for RE retrieval.

## Decision
**OpenAI text-embedding-3-small** via the `openai` Python SDK.
- Dimensions: 1536 (default) — reduce to 512 for cost/speed with negligible quality loss
- Cost: ~$0.02 per 1M tokens — essentially free at ORACLE's signal volume
- Quality: best-in-class for short-text semantic matching (news headlines, market questions)
- Latency: <200ms per batch call with up to 2048 inputs per request

API key stored as `OPENAI_API_KEY` environment variable.

Embedding calls are batched: OSFE accumulates signals for up to 500ms before sending a batch, balancing latency vs cost.

## Consequences
- OSFE and KBPM both use `openai.embeddings.create(model="text-embedding-3-small")`
- Dimensions stored in ChromaDB (ADR-013): 512 (reduced via `dimensions` param)
- If OpenAI API is unavailable, OSFE pauses signal processing and retries — does not crash

## Alternatives Considered
- text-embedding-3-large: 3x cost, marginal improvement for this use case
- Local model via Ollama (nomic-embed-text): no API cost, ~50ms local latency, more ops complexity
- Cohere embed-v3: competitive quality, another API key to manage
