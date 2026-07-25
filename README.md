# Multimodal AI Research & Teaching Assistant

Upload a research paper PDF, index it locally, and get answers grounded in the
document with exact page citations — no cloud API required. The system runs
entirely on your machine using Ollama for language models and FAISS for vector
search. This repository is also a 10-notebook tutorial series covering every
component end-to-end, from PDF ingestion to evaluation.

## Features

- Upload a PDF and build a searchable FAISS index
- Ask questions and receive answers with page citations
- Five teaching modes: beginner, graduate, interview prep, quiz generation, and figure explanation
- Optional figure captioning with a vision-language model
- Source-scoped retrieval — Explain figure mode constrains search to the selected document
- Duplicate upload detection — re-uploading the same PDF returns a cached response instantly
- OpenTelemetry tracing — per-request spans with retrieval scores, token counts, and latency
- Fully local: Ollama + Hugging Face, no API keys required
- Production-style architecture with typed modules, API/UI separation, Docker, testing, CI, evaluation, and observability

## Prerequisites

**Required:**

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Ollama](https://ollama.com)
- Git

Text model:

```bash
ollama pull llama3.2:latest
```

**Optional** — enables figure and image captioning (~6 GB):

```bash
ollama pull qwen2.5vl:latest
```

The vision model is not required for text-only PDF question answering. When it
is not installed, the **Explain figure** mode falls back to text-based
retrieval and shows an in-app prompt with the install command.

## Quick start

```bash
cp .env.example .env
ollama pull llama3.2:3b
docker compose up --build
```

Optional — enable figure captioning:

```bash
ollama pull qwen2.5vl:7b
```

Open:

- UI: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>

**Demo workflow:**

1. In the sidebar, upload `data/sample/attention_is_all_you_need.pdf`
2. Click **Index document**
3. Ask: *"What problem does self-attention solve?"*

## Python package

The core library is distributed as `mrta-rag` on PyPI:

```bash
# Core only (config, schemas, LLM client, prompts)
pip install mrta-rag

# Add PDF ingestion
pip install "mrta-rag[pdf]"

# Add chunking, embeddings, and FAISS vector search
pip install "mrta-rag[retrieval]"

# Full install (matches the Docker environment)
pip install "mrta-rag[all]"
```

```python
import mrta

print(mrta.__version__)   # 0.1.0

# Core API available after pip install mrta-rag:
from mrta import rag_query, LLMClient, Settings, load_prompt

# Requires mrta-rag[pdf]:
from mrta import load_pdf, chunk_pdf

# Requires mrta-rag[retrieval]:
from mrta import Embedder, VectorStore
```

## Development

### Local setup (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

Run the backend and frontend in separate terminals:

```bash
uvicorn apps.api.main:app --reload --port 8000
streamlit run apps/streamlit/app.py
```

### Environment switching

Config is loaded from `configs/{MRTA_ENV}.yaml`, with env vars and `.env`
taking priority:

```bash
MRTA_ENV=test pytest   # lighter models, fast CI
MRTA_ENV=dev pytest    # full dev config
```

### Tests

```bash
pytest
pytest tests/unit/        # unit tests only
pytest tests/evaluation/  # retrieval gate tests
```

### Observability

Tracing is controlled by three `.env` variables:

```bash
ENABLE_TRACING=true           # activate the OTEL SDK
OTEL_CONSOLE_EXPORTER=true    # print spans to stdout (local dev)
OTEL_SERVICE_NAME=mrta
OTEL_EXPORTER_OTLP_ENDPOINT=  # set to export to Jaeger / Tempo
```

With console export enabled, each `/ask` call prints a span to the API logs showing
retrieval scores, cited sources, token counts, and end-to-end latency.

### Linting and type checking

```bash
ruff check src/ tests/ apps/
black --check src/ tests/ apps/
.venv311/bin/mypy src/ apps/ --ignore-missing-imports
```

> **Note:** Use a Python 3.11 virtual environment for `mypy`. The default
> `.venv` uses Python 3.14, whose NumPy stubs use syntax that mypy rejects
> when `python_version = "3.11"` is set. CI uses Python 3.11 and passes.

### Tutorial notebooks

```bash
jupyter lab notebooks/
```

Two parallel versions of the 10-part series:

- **`notebooks/production/`** — imports from `src/mrta/`; the reference implementation
- **`notebooks/tutorials/`** — every function defined inline; use for learning

| # | Phase | Topic |
|---|-------|-------|
| 0 | Setup | Repo scaffold, Ollama, Hugging Face |
| 1 | Ingestion | PyMuPDF text and image extraction |
| 2 | Chunking | Fixed, recursive, and semantic strategies |
| 3 | Embeddings | sentence-transformers + FAISS index |
| 4 | RAG | End-to-end pipeline with citations |
| 5 | Backend | FastAPI endpoints and Pydantic schemas |
| 6 | Frontend | Streamlit upload, ask, cite |
| 7 | Multimodal | Figure extraction, CLIP, VLM captioning |
| 8 | Teaching modes | Prompt templates for different audiences |
| 9 | Evaluation | DeepEval metrics, structured logs, Docker |

### Architecture and design decisions

- Tech stack, system diagram, and repo layout: [`docs/architecture/overview.md`](docs/architecture/overview.md)
- Key design decisions (FAISS vs Qdrant, Ollama vs API, etc.): [`docs/adr/`](docs/adr/)

## Limitations

- Math is rendered as text; LaTeX-aware parsing would improve recall on equation-heavy papers.
- Table extraction is basic; ColPali or `unstructured` would help for table-heavy domains.
- Reranking is a stub; adding a cross-encoder (`bge-reranker-base`) is a one-day improvement.
- No multi-document graph reasoning yet — a clear next step toward an "agentic" research assistant.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
