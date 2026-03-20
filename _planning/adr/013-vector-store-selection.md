# ADR-013: Vector Store for Semantic Similarity

## Status
accepted

## Context
Two use cases: (1) OSFE uses it for signal-to-market similarity search across ~500–5000 active markets. (2) RE uses it to search the /theses/ vault for historical analogues. Both require persistence across restarts.

## Decision
**ChromaDB** — local, file-persisted, Python-native.
- No external service required — runs in-process
- Persistence: file-backed at a configurable `CHROMA_PERSIST_DIR` path
- Two collections: `oracle_markets` (active Polymarket market embeddings) and `oracle_theses` (KBPM thesis embeddings)
- Query latency: <10ms for 10k vectors — more than sufficient
- Python API: `chromadb.PersistentClient`

## Consequences
- OSFE owns `oracle_markets` collection: add markets when SIL surfaces new ones, delete on resolution
- KBPM owns `oracle_theses` collection: add entry per thesis, never delete
- RE queries `oracle_theses` read-only via ChromaDB client
- ChromaDB persist directory must be backed up alongside the KBPM vault

## Alternatives Considered
- FAISS: faster at 1M+ vectors, no built-in persistence layer, more complex API
- Pinecone: managed, paid, unnecessary external dependency at this scale
- Weaviate: full-featured, significantly more ops overhead than needed
