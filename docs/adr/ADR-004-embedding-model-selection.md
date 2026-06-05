# ADR-004 — Embedding Model Selection

**Status:** Accepted  
**Date:** 2026-06-04

---

## Context

Embedding quality directly affects retrieval quality, which is the single largest driver of RAG answer quality. Two embedding modalities are needed:

1. **Text embeddings** — for chunked PDF text.
2. **Image embeddings** — for figures, diagrams, slides (multimodal retrieval).

Constraints:

- Must run locally without API keys.
- Must be fast enough for tutorial use (no 30-second waits per query).
- Should demonstrate awareness of quality tradeoffs (small vs large models).

Text models evaluated: `all-MiniLM-L6-v2` (22M params), `nomic-embed-text` (via Ollama), `BGE-large-en-v1.5` (335M params).  
Image models evaluated: CLIP `ViT-B/32`, CLIP `ViT-L/14`.

## Decision

**Two-tier text embedding strategy; CLIP for images.**

| Environment | Text embedding | Why |
|---|---|---|
| `test` / CI | `sentence-transformers/all-MiniLM-L6-v2` | Fast CPU inference, no Ollama needed |
| `dev` (tutorial) | `nomic-embed-text` (Ollama) | Higher quality, same Ollama server used for LLM |
| Production (optional) | `BGE-large-en-v1.5` | Best open-source quality; documented as upgrade path |

Image: `openai/clip-vit-base-patch32` (default). Upgrade path: `ViT-L-14` for better image-text alignment.

Configured via `settings.embedding_model` and `settings.clip_model` in `configs/{env}.yaml`. No code changes needed to swap.

## Consequences

**Positive:**

- CI runs without Ollama (uses MiniLM from sentence-transformers).
- Dev tutorials use nomic-embed-text, which performs better on long-form text.
- BGE upgrade is documented and config-only.

**Negative / Tradeoffs:**

- nomic-embed-text requires Ollama to be running; documented in setup.
- CLIP ViT-B/32 is weaker than ViT-L/14 for fine-grained figure retrieval; acceptable for tutorials.
- Embedding dimension mismatch if model is swapped after index is built — requires re-indexing.

## References

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text)
- [BGE models (BAAI)](https://huggingface.co/BAAI)
- [CLIP paper](https://arxiv.org/abs/2103.00020)
