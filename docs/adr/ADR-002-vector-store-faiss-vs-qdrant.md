# ADR-002 — Vector Store: FAISS vs Qdrant

**Status:** Accepted  
**Date:** 2026-06-04

---

## Context

A vector store is required to index and retrieve document chunk embeddings. Two options were evaluated:

| | FAISS | Qdrant |
|---|---|---|
| Deployment | In-process Python | Docker container |
| Persistence | Manual save/load to disk | Native persistence |
| Metadata filtering | Not natively supported | First-class support |
| Scale | Single-machine, RAM-bound | Distributed, production-ready |
| Setup for tutorials | Zero infra | Requires Docker |

The project has two simultaneous goals: tutorial simplicity (students run everything locally) and production credibility (demonstrating awareness of real-world tradeoffs).

## Decision

**Default to FAISS; provide a drop-in Qdrant swap.**

- `settings.vector_store` (in `configs/dev.yaml`) defaults to `"faiss"`.
- The retrieval interface (`src/mrta/retrieval/vector_store.py`) is identical for both backends — swap is one config change: `vector_store: qdrant`.
- Qdrant is demonstrated in Notebook 09 with a Docker-based setup.

This mirrors how production systems are actually introduced: start simple, swap the backend without touching retrieval logic.

## Consequences

**Positive:**
- Students can run Notebooks 00–08 with zero Docker.
- Qdrant demo in Notebook 09 shows the production swap concretely.
- Interface-based design means neither backend leaks into RAG pipeline logic.

**Negative / Tradeoffs:**
- FAISS does not support metadata filtering natively; workaround is post-retrieval filtering in Python.
- Two code paths to maintain in `vector_store.py`.

## References

- [FAISS documentation](https://faiss.ai/)
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [Notebook 09 — Evaluation, Logging, Docker](../../notebooks/tutorials/2026-05-25-phase09-evaluation-logging-docker.ipynb)
