# Architecture overview

## System diagram

```text
PDFs / Slides / Images
        ↓
   Ingestion (PyMuPDF + pdfplumber)
        ↓
   Chunking (page-aware, metadata-preserving)
        ↓
   Embeddings (sentence-transformers / BGE)   +   Image embeddings (CLIP)
        ↓                                                  ↓
            Vector store (FAISS  ↔  Qdrant)
        ↓
   Retrieval + optional reranker
        ↓
   LLM (Ollama / HF)  +  VLM (qwen2.5vl) for figures
        ↓
   Grounded answer with citations
        ↓
   FastAPI backend  →  Streamlit UI
        ↑
   Logs + DeepEval evaluation + Docker
```

## Tech stack

| Layer            | Choice                                                                 | Why                                                                                  |
| ---------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| PDF parsing      | PyMuPDF (`fitz`), pdfplumber                                           | Fast, page-accurate, exposes images and layout boxes.                                |
| Chunking         | LangChain `RecursiveCharacterTextSplitter` + custom page-aware wrapper | Token-aware, preserves `{doc_id, page, section}` metadata.                           |
| Text embeddings  | `sentence-transformers/all-MiniLM-L6-v2` (default), BGE-large optional | Free, runs on CPU; BGE for higher quality.                                           |
| Image embeddings | CLIP (`openai/clip-vit-base-patch32`)                                  | Shared text-image space, standard for multimodal retrieval.                          |
| Vector store     | FAISS (default), Qdrant (Docker)                                       | FAISS for tutorial simplicity, Qdrant for the production swap demo.                  |
| LLM              | Ollama (`llama3.2`, `mistral`, `qwen2.5`)                              | Local, no API key, swappable; HF Transformers fallback supported.                    |
| VLM              | Ollama `qwen2.5vl:7b` / HF `Qwen2-VL-2B-Instruct`                     | Open-source, runnable on consumer hardware.                                          |
| Backend          | FastAPI + Pydantic v2                                                  | Typed, async-ready, auto-docs at `/docs`.                                            |
| Frontend         | Streamlit                                                              | Single-file UI, fast iteration.                                                      |
| Evaluation       | DeepEval                                                               | Groundedness, faithfulness, context precision; CI-friendly assertions.               |
| Observability    | Structured JSONL logs + OpenTelemetry hooks                            | Per-run traceability, low overhead.                                                  |
| Infra            | Docker + Compose (`compose.yaml`)                                      | Reproducible local and production environment.                                       |
| CI               | GitHub Actions (ruff, black, pytest)                                   | Standard senior-engineer hygiene.                                                    |

## Repo layout

```text
multimodal-research-teaching-assistant/
├── README.md
├── CLAUDE.md
├── RESEARCH_NOTES.md
├── pyproject.toml
├── requirements.txt
├── compose.yaml              ← Docker entry point (run from repo root)
├── .env.example              ← copy to .env before first run
├── src/
│   └── mrta/                 installable Python library (pip install -e .)
│       ├── core/             config.py, schemas.py, llm.py, rag_pipeline.py
│       ├── ingestion/        pdf_loader.py, chunker.py
│       ├── retrieval/        embedder.py, vector_store.py
│       ├── multimodal/       clip_embedder.py, vlm_client.py
│       ├── evaluation/
│       ├── generation/
│       ├── prompts/          Jinja2 prompt templates
│       └── observability/    logging.py, tracing.py
├── apps/
│   ├── api/                  FastAPI entry point (imports from mrta.*)
│   └── streamlit/            Streamlit UI
├── configs/                  dev.yaml, test.yaml — environment-specific defaults
├── data/
│   ├── sample/               demo PDF (Attention Is All You Need)
│   ├── golden_qa.yaml        hand-labeled benchmark for evaluation
│   ├── uploads/              runtime — user-uploaded files (gitignored)
│   ├── logs/                 runtime — structured JSONL logs (gitignored)
│   └── vector_store/         runtime — FAISS indexes (gitignored)
├── examples/                 standalone usage examples
├── notebooks/
│   ├── tutorials/            original series (00–09) — inline implementations
│   └── production/           same series — imports from mrta.* library
├── tests/                    pytest unit + integration + fixtures
├── docker/                   Dockerfile.api, Dockerfile.streamlit
├── docs/                     adr/, architecture/, observability.md
└── .github/workflows/        ci.yml
```

## Design tradeoffs

- **FAISS vs Qdrant.** FAISS is in-process, zero infra, and perfect for the tutorial. Qdrant is swapped in via Docker when persistence and metadata filtering are needed. The vector store interface (`src/mrta/retrieval/vector_store.py`) is identical for both.
- **Local models vs API.** Defaulting to Ollama keeps the project free, private, and reproducible. The LLM client (`src/mrta/core/llm.py`) is provider-agnostic — switching to OpenAI is one config change.
- **Chunk size.** 500–800 tokens with 100-token overlap is the sweet spot for research papers; Notebook 02 shows a side-by-side recall comparison across strategies.
- **Citations.** Every chunk stores `{doc_id, page, section, chunk_id}` so the LLM can quote exact pages, and cited pages are re-verified in the evaluation pipeline.
