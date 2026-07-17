# Research Notes

Living document for paper summaries, experiment observations, open questions, and implementation notes.

---

## Papers

### Attention Is All You Need (Vaswani et al., 2017)

- **Why relevant:** Primary benchmark document for the RAG pipeline.
- **Key ideas:** Transformer architecture, scaled dot-product attention, multi-head attention.
- **RAG notes:** Dense mathematical notation; embedding quality matters more than chunk size for retrieval on this paper. Equation references (e.g., "Equation 1") require the chunk to include surrounding prose for context.
- **Citation:** Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. [arxiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

### RAG — Retrieval-Augmented Generation (Lewis et al., 2020)

- **Why relevant:** Foundational paper for this project's retrieval architecture.
- **Key ideas:** Non-parametric memory via retrieval, two-step seq2seq generation.
- **Implementation notes:** We use a simpler "retrieve → generate" pipeline (no joint fine-tuning). The paper's DPR retriever is replaced by sentence-transformers + FAISS.
- **Citation:** Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. [arxiv:2005.11401](https://arxiv.org/abs/2005.11401)

---

### Lost in the Middle (Liu et al., 2023)

- **Why relevant:** Explains why top-k > 5 often hurts RAG performance.
- **Key ideas:** LLMs attend poorly to information in the middle of long contexts. Best performance when relevant info is at beginning or end of context.
- **Implementation notes:** We default to top_k=5 partly because of this finding. Reranker (when added) should push highest-quality chunks to positions 0 and k-1.
- **Citation:** Liu, N.F. et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. [arxiv:2307.03172](https://arxiv.org/abs/2307.03172)

---

## Experiment Log

### 2026-06-04 — Baseline config validation

- `MRTA_ENV=dev` loads `configs/dev.yaml`: nomic-embed-text, chunk_size=1000, DEBUG logging. ✅
- `MRTA_ENV=test` loads `configs/test.yaml`: MiniLM, chunk_size=500, WARNING logging. ✅
- Priority chain verified: env var overrides YAML. ✅

---

## Open Questions

- [ ] Does page-boundary chunking lose too much context for equations that span pages?
- [ ] What is the retrieval quality gap between `all-MiniLM-L6-v2` and `nomic-embed-text` on our benchmark PDF?
- [ ] Is `top_k=5` the right default? Run ablation in Notebook 04.
- [ ] Can CLIP retrieve figures reliably when query is text-only? Needs multimodal benchmark.
- [ ] Does `qwen2.5vl:7b` produce better figure captions than `llava:7b` for diagram-heavy papers?

---

## Implementation Notes

### Chunking strategy (Notebook 02)

- LangChain `RecursiveCharacterTextSplitter` splits on `["\n\n", "\n", " ", ""]` in order.
- `chunk_size` is measured in characters by default — set `length_function=len` explicitly to avoid surprises.
- Metadata attached per chunk: `{doc_id, page, source, chunk_id}` — `page` is critical for citation.

### FAISS index lifecycle

- Build: `faiss.IndexFlatL2` for exact search (tutorial simplicity).
- Save: `faiss.write_index(index, path)` + pickle metadata dict.
- Load: `faiss.read_index(path)`.
- Note: FAISS index must be rebuilt if embedding model changes (dimension mismatch).

---

## References

- [MTEB Leaderboard (embedding benchmarks)](https://huggingface.co/spaces/mteb/leaderboard)
- [Ollama model library](https://ollama.com/library)
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/)
- ADRs: [docs/adr/](docs/adr/)
