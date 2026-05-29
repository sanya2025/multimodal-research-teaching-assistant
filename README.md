# Multimodal AI Research & Teaching Assistant

A production-style multimodal AI system that ingests PDFs, slides, diagrams, and images, and produces grounded answers, figure explanations, summaries, quizzes, and teaching plans — all with citations back to the source.

> **MVP promise:** Upload a research paper PDF, ask questions, get answers grounded in the paper with page citations. Then explain figures with a vision-language model.

This repo is the flagship portfolio project paired with a 10-notebook tutorial series in [`notebooks/`](notebooks/) that walks through every phase end-to-end on local open-source models (Ollama + Hugging Face) — no API keys required.

---

## Table of Contents

- [Multimodal AI Research \& Teaching Assistant](#multimodal-ai-research--teaching-assistant)
  - [Table of Contents](#table-of-contents)
  - [Problem statement](#problem-statement)
  - [Architecture](#architecture)
  - [Tech stack](#tech-stack)
  - [Setup](#setup)
  - [Quick start](#quick-start)
  - [Repo layout](#repo-layout)
  - [Tutorial notebooks](#tutorial-notebooks)
  - [Evaluation](#evaluation)
  - [Design tradeoffs](#design-tradeoffs)
  - [Limitations \& future work](#limitations--future-work)
  - [License](#license)

---

## Problem statement

Researchers, graduate students, and engineers regularly need to read and reason over PDFs that mix prose, math, diagrams, and code. General-purpose chatbots hallucinate, lose page context, and cannot reliably reason over figures. This project builds a *grounded* assistant: every answer is traced to the page (and ideally the figure) it came from.

## Architecture

```
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
   LLM (Ollama / HF)  +  VLM (LLaVA / Qwen2-VL) for figures
        ↓
   Grounded answer with citations
        ↓
   FastAPI backend  →  Streamlit UI
        ↑
   Logs + Ragas evaluation + Docker
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
| VLM              | Ollama `llava` / HF `Qwen2-VL-2B-Instruct`                             | Open-source, runnable on consumer hardware.                                          |
| Backend          | FastAPI + Pydantic v2                                                  | Typed, async-ready, auto-docs at `/docs`.                                            |
| Frontend         | Streamlit                                                              | Single-file UI, fast iteration.                                                      |
| Evaluation       | Ragas, DeepEval                                                        | Groundedness, faithfulness, context precision; CI-friendly assertions.               |
| Observability    | Structured JSONL logs + OpenTelemetry hooks                            | Per-run traceability, low overhead.                                                  |
| Infra            | Docker + docker-compose                                                | Reproducible local + production environment.                                         |
| CI               | GitHub Actions (ruff, black, pytest)                                   | Standard senior-engineer hygiene.                                                    |

## Setup

```bash
git clone <your-fork-url>
cd multimodal-research-teaching-assistant

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env

# Install Ollama (https://ollama.com) then pull the default models:
ollama pull llama3.2:3b
ollama pull llava:7b
ollama pull nomic-embed-text   # optional alternative embedder
```

## Quick start

```bash
# 1. Walk through the tutorials end-to-end
jupyter lab notebooks/

# 2. Or run the app:
uvicorn app.api.main:app --reload --port 8000     # backend
streamlit run frontend/streamlit_app.py           # frontend (separate terminal)
```

Upload a paper, ask a question, see the answer with page citations.

## Repo layout

```
multimodal-research-teaching-assistant/
├── app/
│   ├── api/              FastAPI routers + main.py
│   ├── core/             rag_pipeline.py, llm.py, config.py
|   |--- generation/ 
│   ├── ingestion/        pdf_loader.py, chunker.py, figure_extractor.py
│   ├── retrieval/        vector_store.py, embedder.py, reranker.py
│   ├── multimodal/       vlm_client.py, clip_embedder.py
│   ├── evaluation/       eval_pipeline.py, metrics.py
│   ├── prompts/          Jinja2 prompt templates
│   └── observability/    logging.py, tracing.py
├── frontend/streamlit_app.py
├── data/{raw,processed,vector_store,logs}/
├── notebooks/            Tutorial series (00 → 09)
├── tests/                pytest unit + integration
├── configs/              YAML configs per env
├── docker/               Dockerfile.api, Dockerfile.ui, docker-compose.yml
├── docs/                 architecture.md, system_design.md, evaluation.md
└── .github/workflows/    ci.yml
```

## Tutorial notebooks

The `notebooks/` folder contains a 10-part tutorial that builds the system from scratch, fully runnable on local models:

| # | Notebook                                                | What you learn                                  |
|---|---------------------------------------------------------|-------------------------------------------------|
| 0 | `2026-05-25-phase00-foundations-and-setup.ipynb`        | Repo scaffold, venv, Ollama, HF setup           |
| 1 | `2026-05-25-phase01-pdf-ingestion.ipynb`                | PyMuPDF text/image extraction, doc schema       |
| 2 | `2026-05-25-phase02-chunking-strategies.ipynb`          | Fixed/recursive/semantic chunking + metadata    |
| 3 | `2026-05-25-phase03-embeddings-and-faiss.ipynb`         | sentence-transformers + FAISS index lifecycle   |
| 4 | `2026-05-25-phase04-rag-pipeline.ipynb`                 | End-to-end RAG with Ollama + citations          |
| 5 | `2026-05-25-phase05-fastapi-backend.ipynb`              | Endpoints, Pydantic schemas, async patterns     |
| 6 | `2026-05-25-phase06-streamlit-frontend.ipynb`           | UI walkthrough, upload/ask/cite flow            |
| 7 | `2026-05-25-phase07-figure-extraction-and-vlm.ipynb`    | Figure extraction, CLIP, LLaVA captioning       |
| 8 | `2026-05-25-phase08-teaching-modes-and-prompts.ipynb`   | Beginner/grad/interview/quiz prompt patterns    |
| 9 | `2026-05-25-phase09-evaluation-logging-docker.ipynb`    | Ragas, structured logs, Docker, README polish   |

## Evaluation

We track six metrics per question:

- **Answer relevance** — does the answer address the question?
- **Groundedness** — is every claim supported by retrieved context?
- **Citation correctness** — do cited pages actually contain the claim?
- **Retrieval precision** — fraction of retrieved chunks that were useful.
- **Latency** — end-to-end seconds, p50/p95.
- **Hallucination rate** — claims not in any retrieved chunk.

Notebook 09 builds the eval pipeline with Ragas and a small hand-labeled benchmark.

## Design tradeoffs

- **FAISS vs Qdrant.** FAISS is in-process, zero infra, and perfect for the tutorial. We swap to Qdrant in Docker once we want persistence + filtering on metadata. The vector store interface (`app/retrieval/vector_store.py`) is identical for both.
- **Local models vs API.** Defaulting to Ollama keeps the project free, private, and reproducible. The LLM client (`app/core/llm.py`) is provider-agnostic — flipping to OpenAI is one config change.
- **Chunk size.** 500–800 tokens with 100-token overlap is the sweet spot for research papers; we revisit this in Notebook 02 with a side-by-side recall comparison.
- **Citations.** We store `{doc_id, page, section, chunk_id}` on every chunk so the LLM can quote exact pages, and we re-verify cited pages in the eval pipeline.

## Limitations & future work

- Math is rendered as text; LaTeX-aware parsing would improve recall on equation-heavy papers.
- Table extraction is basic; ColPali or `unstructured` would help for table-heavy domains.
- Reranking is a stub; adding a cross-encoder (`bge-reranker-base`) is a one-day improvement.
- No multi-document graph reasoning yet — a clear next step toward an "agentic" research assistant.

---

Made for portfolio review, interview discussion, and actually being useful when reading papers.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
