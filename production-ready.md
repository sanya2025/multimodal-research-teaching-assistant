# Production-Ready Plan

This document maps every tutorial notebook to the `src/mrta/` library.  
It answers three questions for each notebook: **what to extract**, **where it lives**, and **why**.

---

## Guiding Principle

Notebooks are for teaching. The library is for production.

The workflow for every notebook is the same:

1. **Teach** — write the function inline in the notebook so students can see exactly what it does.
2. **Extract** — move the function into `src/mrta/` with type hints, a one-line docstring, and a test.
3. **Import** — replace the inline definition in the notebook with `from mrta.X import Y`.
4. **Validate** — run `MRTA_ENV=test pytest` to confirm nothing broke.

When this is done for all notebooks, the library is complete and the notebooks become thin wrappers that demonstrate the API.

---

## Library map

```text
src/mrta/
├── core/
│   ├── config.py          Settings (pydantic-settings, YAML-backed)       ✅ done
│   ├── schemas.py         PageRecord, PdfDocument, Chunk                   partial
│   ├── llm.py             LLMClient — provider-agnostic text generation    stub
│   ├── rag_pipeline.py    rag_query() — retrieve → prompt → generate       stub
│   └── exceptions.py      MrtaError base + subclasses                      stub
├── ingestion/
│   ├── pdf_loader.py      load_pdf(), _doc_id()                            ✅ done
│   ├── chunker.py         fixed_chunks(), recursive_chunks(),
│   │                      token_chunks(), semantic_chunks()                 stub
│   └── figure_extractor.py extract_figures() → list[FigureRecord]          stub
├── retrieval/
│   ├── embedder.py        Embedder (sentence-transformers + Ollama)        stub
│   ├── vector_store.py    VectorStore (FAISS default, Qdrant swap)         stub
│   └── reranker.py        Reranker (cross-encoder, optional)               stub
├── multimodal/
│   ├── clip_embedder.py   CLIPEmbedder — image → float32 vector            stub
│   └── vlm_client.py      VLMClient — image + text → caption               stub
├── prompts/
│   ├── __init__.py        load_prompt(name, **kwargs) → str                stub
│   ├── rag.j2             base RAG template                                stub
│   ├── explain.j2         explain a figure or concept                      stub
│   ├── quiz.j2            generate quiz questions                          stub
│   ├── beginner.j2        simplified explanation                           stub
│   └── expert.j2          grad-student depth                               stub
├── evaluation/
│   ├── eval_pipeline.py   run_eval(benchmark) → EvalReport                 stub
│   └── metrics.py         answer_relevance(), faithfulness(),
│                          citation_correctness(), hallucination_rate()     stub
└── observability/
    ├── logging.py         StructuredLogger — JSONL per run                 stub
    └── tracing.py         OpenTelemetry spans (optional)                   stub
```

---

## Phase 00 — Foundations & Setup

**Notebook teaches:** virtual environment, Ollama health-check, first LLM call, embedding shape, import sanity check, loading `mrta.core.config.settings`.

**What to extract:** nothing new — this notebook is purely environment setup.

**What the notebook should import from the library:**

```python
from mrta.core.config import settings
print(settings.llm_provider, settings.embedding_model)
```

**Done when:** the notebook passes its own import sanity cell with `mrta` installed via `pip install -e ".[all]"`.

---

## Phase 01 — PDF Ingestion

**Notebook teaches:** PyMuPDF structure, document schema design, `load_pdf`, image extraction, OCR fallback.

**What to extract:**

| Function / class | Target file | Why |
|-----------------|-------------|-----|
| `PageRecord`, `PdfDocument` | `src/mrta/core/schemas.py` | Shared across ingestion, retrieval, API — must not be re-defined per notebook |
| `load_pdf(path) -> PdfDocument` | `src/mrta/ingestion/pdf_loader.py` | ✅ already done |
| `extract_figures(path) -> list[FigureRecord]` | `src/mrta/ingestion/figure_extractor.py` | Used again in Phase 07; single source of truth |
| `ocr_page_if_needed(page) -> str` | `src/mrta/ingestion/pdf_loader.py` | Called inside `load_pdf` when text is empty |

**`FigureRecord` schema to add to `schemas.py`:**

```python
class FigureRecord(BaseModel):
    doc_id: str
    page: int
    figure_index: int          # 1-indexed per page
    image_bytes: bytes
    source: str
```

**Notebook after extraction:**

```python
from mrta.ingestion import load_pdf
from mrta.ingestion.figure_extractor import extract_figures

pdf = load_pdf("data/sample/attention_is_all_you_need.pdf")
figures = extract_figures("data/sample/attention_is_all_you_need.pdf")
```

**Tests to write:** `tests/unit/test_figure_extractor.py` — assert figure count, image bytes non-empty.

---

## Phase 02 — Chunking Strategies

**Notebook teaches:** fixed-size, recursive, token-aware, and semantic chunking; side-by-side comparison of chunk quality; saving chunks to `data/processed/chunks.jsonl`.

**What to extract:**

| Function | Target file | Why |
|----------|-------------|-----|
| `Chunk` schema | `src/mrta/core/schemas.py` | Used by retrieval, RAG, API, evaluation |
| `fixed_chunks(pdf, size, overlap)` | `src/mrta/ingestion/chunker.py` | Baseline strategy |
| `recursive_chunks(pdf, size, overlap)` | `src/mrta/ingestion/chunker.py` | Default production strategy |
| `token_chunks(pdf, size, overlap)` | `src/mrta/ingestion/chunker.py` | Token-budget-aware; needed for accurate context window math |
| `semantic_chunks(pdf, threshold)` | `src/mrta/ingestion/chunker.py` | Advanced; slow but highest quality |
| `chunk_pdf(pdf, strategy, **kwargs)` | `src/mrta/ingestion/chunker.py` | Single entry point used by the RAG pipeline |

**`Chunk` schema to add to `schemas.py`:**

```python
class Chunk(BaseModel):
    chunk_id: str          # "{doc_id}_p{page}_c{idx}"
    doc_id: str
    source: str
    page: int
    text: str
    section: str | None = None
    n_tokens: int | None = None
```

**`chunk_pdf` is the production entry point** — it dispatches to the right strategy based on the `chunking_strategy` setting in `configs/dev.yaml`:

```python
def chunk_pdf(
    pdf: PdfDocument,
    strategy: str = "recursive",
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]: ...
```

**Notebook after extraction:**

```python
from mrta.ingestion.chunker import chunk_pdf
chunks = chunk_pdf(pdf, strategy="recursive")
```

**Tests to write:** `tests/unit/test_chunker.py` — load fixture PDF, assert chunk count > 0, chunk IDs unique, every chunk has `doc_id` and `page`.

---

## Phase 03 — Embeddings & FAISS

**Notebook teaches:** sentence-transformers embedding pipeline, FAISS index lifecycle, cosine similarity search, wrapping into a reusable `VectorStore`.

**What to extract:**

| Class / function | Target file | Why |
|-----------------|-------------|-----|
| `Embedder` | `src/mrta/retrieval/embedder.py` | Swappable between sentence-transformers and Ollama; config-driven model selection |
| `VectorStore` | `src/mrta/retrieval/vector_store.py` | Used by RAG pipeline, FastAPI, and evaluation; must persist to disk and reload |

**`Embedder` interface:**

```python
class Embedder:
    def __init__(self, model_name: str | None = None) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...   # float32, L2-normalised
    @property
    def dim(self) -> int: ...
```

Model selection reads from `settings.embedding_model` by default; `model_name` overrides it. This is why `configs/test.yaml` uses `all-MiniLM-L6-v2` (no Ollama) and `configs/dev.yaml` uses `nomic-embed-text` — the same class handles both.

**`VectorStore` interface:**

```python
class VectorStore:
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, k: int = 5) -> list[Chunk]: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "VectorStore": ...
```

`save` / `load` write two files: `index.faiss` and `metadata.jsonl`. This is the swap boundary for Qdrant (ADR-002) — the Qdrant implementation exposes the same interface.

**Notebook after extraction:**

```python
from mrta.retrieval.embedder import Embedder
from mrta.retrieval.vector_store import VectorStore

embedder = Embedder()
vs = VectorStore(embedder)
vs.add(chunks)
results = vs.search("What is multi-head attention?", k=5)
```

**Tests to write:** `tests/unit/test_vector_store.py` — build index from fixture chunks, search returns k results, save/load round-trip preserves chunk text.

---

## Phase 04 — End-to-End RAG Pipeline

**Notebook teaches:** prompt templating, LLM client wrapper, full retrieve → prompt → generate cycle, structured logging of every run.

**What to extract:**

| Class / function | Target file | Why |
|-----------------|-------------|-----|
| `LLMClient` | `src/mrta/core/llm.py` | Provider-agnostic; reads `settings.llm_provider` and dispatches to Ollama / HF / OpenAI |
| `rag_query(question, vs, llm)` | `src/mrta/core/rag_pipeline.py` | The single function called by the FastAPI route — must be independently testable |
| RAG prompt template | `src/mrta/prompts/rag.j2` | Decoupled from Python code; editable without touching logic |
| `StructuredLogger` | `src/mrta/observability/logging.py` | Every run appended to `data/logs/runs.jsonl` as one JSON line |

**`LLMClient` interface:**

```python
class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None: ...
    def chat(self, messages: list[dict], temperature: float = 0.1) -> str: ...
```

**`rag_query` signature:**

```python
def rag_query(
    question: str,
    vector_store: VectorStore,
    llm: LLMClient,
    top_k: int = 5,
) -> dict:
    """Returns {"answer": str, "sources": list[Chunk], "latency_s": float}"""
```

**RAG prompt template (`src/mrta/prompts/rag.j2`):**

```
You are a research assistant. Answer the question using only the context below.
Cite the page number(s) you drew from.

Context:
{% for chunk in chunks %}
[Page {{ chunk.page }}] {{ chunk.text }}
{% endfor %}

Question: {{ question }}
Answer:
```

**Notebook after extraction:**

```python
from mrta.core.llm import LLMClient
from mrta.core.rag_pipeline import rag_query

llm = LLMClient()
result = rag_query("What problem does attention solve?", vs, llm)
print(result["answer"])
print("Sources:", [c.page for c in result["sources"]])
```

**Tests to write:** `tests/unit/test_rag_pipeline.py` — mock `LLMClient.chat`, assert `rag_query` returns a dict with `answer` and `sources` keys; assert sources come from the vector store.

---

## Phase 05 — FastAPI Backend

**Notebook teaches:** Pydantic request/response schemas, FastAPI route design, async patterns, driving the API from a notebook.

**What to extract:**

| Item | Target file | Why |
|------|-------------|-----|
| `AskRequest`, `AskResponse` | `apps/api/schemas/ask.py` | Pydantic v2 schemas; versioned independently of the library |
| `UploadResponse` | `apps/api/schemas/upload.py` | Separating schemas by endpoint keeps each small and testable |
| `/ask` route | `apps/api/routers/ask.py` | Routes import from `mrta.*`; they do not re-implement logic |
| `/upload` route | `apps/api/routers/upload.py` | Saves PDF to `data/raw/`, calls `load_pdf`, stores in a registry |
| `/documents` route | `apps/api/routers/documents.py` | Lists ingested doc IDs |

**Route design rule:** every FastAPI route is a thin adapter. It validates input, calls one `mrta.*` function, and returns the result. No business logic in routes.

```python
# apps/api/routers/ask.py
@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    result = rag_query(req.question, get_vector_store(), get_llm(), top_k=req.top_k)
    return AskResponse(answer=result["answer"], sources=result["sources"])
```

**Tests to write:** `tests/unit/test_api.py` — use `httpx.AsyncClient` with `TestClient`, mock `rag_query`, assert 200 + correct response shape.

---

## Phase 06 — Streamlit Frontend

**Notebook teaches:** Streamlit layout, file upload widget, calling the FastAPI backend, displaying answers with page citations.

**What to extract:** nothing — all UI code belongs in `apps/streamlit/app.py`.

**The one rule:** `apps/streamlit/app.py` must not import from `apps.api`. It calls the REST API via `httpx`. This keeps the frontend deployable independently.

```python
# apps/streamlit/app.py
import httpx, streamlit as st

API = "http://localhost:8000"

uploaded = st.file_uploader("Upload a PDF")
if uploaded:
    httpx.post(f"{API}/upload", files={"file": uploaded.getvalue()})

question = st.text_input("Ask a question")
if question:
    resp = httpx.post(f"{API}/ask", json={"question": question})
    data = resp.json()
    st.write(data["answer"])
    for s in data["sources"]:
        st.caption(f"— page {s['page']}, {s['source']}")
```

**Tests to write:** none for Streamlit UI itself; covered by API integration tests.

---

## Phase 07 — Figure Extraction & VLM

**Notebook teaches:** bounding-box-aware figure extraction, CLIP embeddings for cross-modal retrieval, VLM captioning with LLaVA or Qwen2-VL.

**What to extract:**

| Class / function | Target file | Why |
|-----------------|-------------|-----|
| `extract_figures(path) -> list[FigureRecord]` | `src/mrta/ingestion/figure_extractor.py` | Also needed in Phase 01; single implementation |
| `CLIPEmbedder` | `src/mrta/multimodal/clip_embedder.py` | Image embedding; separate from text embedder |
| `VLMClient` | `src/mrta/multimodal/vlm_client.py` | Wraps Ollama LLaVA and HF Qwen2-VL behind one interface |

**`CLIPEmbedder` interface:**

```python
class CLIPEmbedder:
    def embed_image(self, image: Image.Image) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...
```

**`VLMClient` interface:**

```python
class VLMClient:
    def caption(self, image: Image.Image, prompt: str | None = None) -> str: ...
```

**Notebook after extraction:**

```python
from mrta.ingestion.figure_extractor import extract_figures
from mrta.multimodal.clip_embedder import CLIPEmbedder
from mrta.multimodal.vlm_client import VLMClient

figures = extract_figures("data/sample/attention_is_all_you_need.pdf")
clip = CLIPEmbedder()
vlm = VLMClient()
caption = vlm.caption(figures[0].to_pil(), prompt="Describe this figure.")
```

**Tests to write:** `tests/unit/test_clip_embedder.py` — embed a 1×1 white image, assert shape is `(512,)` and norm is ~1.0.

---

## Phase 08 — Teaching Modes & Prompt Engineering

**Notebook teaches:** Jinja2 templating, five teaching-mode prompts (base, beginner, expert, quiz, interview), prompt-quality heuristics.

**What to extract:**

| Item | Target file | Why |
|------|-------------|-----|
| `load_prompt(name, **kwargs) -> str` | `src/mrta/prompts/__init__.py` | Single function; callers never touch Jinja2 directly |
| `rag.j2` | `src/mrta/prompts/rag.j2` | ✅ planned in Phase 04 |
| `beginner.j2` | `src/mrta/prompts/beginner.j2` | Teaching mode |
| `expert.j2` | `src/mrta/prompts/expert.j2` | Teaching mode |
| `quiz.j2` | `src/mrta/prompts/quiz.j2` | Teaching mode |
| `explain.j2` | `src/mrta/prompts/explain.j2` | Figure explanation |

**`load_prompt` implementation:**

```python
from jinja2 import Environment, PackageLoader

_env = Environment(loader=PackageLoader("mrta", "prompts"))

def load_prompt(name: str, **kwargs: object) -> str:
    """Render a Jinja2 template from src/mrta/prompts/{name}.j2."""
    return _env.get_template(f"{name}.j2").render(**kwargs)
```

**Notebook after extraction:**

```python
from mrta.prompts import load_prompt

prompt = load_prompt("beginner", question="What is attention?", chunks=results)
print(llm.chat([{"role": "user", "content": prompt}]))
```

**Tests to write:** `tests/unit/test_prompts.py` — call `load_prompt("rag", question="Q", chunks=[])`, assert returns a non-empty string containing "Q".

---

## Phase 09 — Evaluation, Observability & Docker

**Notebook teaches:** building a benchmark, custom metrics, Ragas for LLM-judged groundedness, structured logging, Docker, docker-compose, Qdrant swap, CI.

**What to extract:**

| Class / function | Target file | Why |
|-----------------|-------------|-----|
| `answer_relevance(question, answer)` | `src/mrta/evaluation/metrics.py` | Deterministic substring/embedding metric; fast, no LLM call |
| `faithfulness(answer, chunks)` | `src/mrta/evaluation/metrics.py` | Checks every claim is grounded in retrieved context |
| `citation_correctness(answer, chunks)` | `src/mrta/evaluation/metrics.py` | Verifies cited page numbers exist in the source chunks |
| `hallucination_rate(answer, chunks)` | `src/mrta/evaluation/metrics.py` | Fraction of sentences with no grounding chunk |
| `run_eval(benchmark, vs, llm)` | `src/mrta/evaluation/eval_pipeline.py` | Runs all metrics over a list of `{question, expected_answer}` pairs |
| `StructuredLogger` | `src/mrta/observability/logging.py` | Appends one JSON line per run to `data/logs/runs.jsonl` |

**`EvalReport` schema to add to `schemas.py`:**

```python
class EvalReport(BaseModel):
    n_questions: int
    answer_relevance: float
    faithfulness: float
    citation_correctness: float
    hallucination_rate: float
    mean_latency_s: float
```

**`StructuredLogger` interface:**

```python
class StructuredLogger:
    def log_run(self, question: str, answer: str, sources: list[Chunk], latency_s: float) -> None:
        """Appends one JSON line to settings.log_file."""
```

**Notebook after extraction:**

```python
from mrta.evaluation.eval_pipeline import run_eval

report = run_eval(benchmark, vs, llm)
print(f"Faithfulness : {report.faithfulness:.2f}")
print(f"Hallucination: {report.hallucination_rate:.2f}")
```

**Tests to write:** `tests/unit/test_metrics.py` — pass a known answer + known chunks, assert metric values are in [0, 1].

---

## Definition of Done

A module is production-ready when:

- [ ] Implementation lives in `src/mrta/`, not inline in a notebook
- [ ] Public symbols exported via `__all__` in the submodule `__init__.py`
- [ ] All arguments and return types annotated
- [ ] One-line docstring on every public function and class
- [ ] At least one unit test in `tests/unit/`
- [ ] `MRTA_ENV=test pytest` passes in under 30 seconds (no Ollama required)
- [ ] `ruff check src/ tests/` and `black --check src/ tests/` pass
- [ ] Notebook imports from `mrta.*` — no inline re-definitions

The project is production-ready when every row in the library map above is checked off.

---

## Extraction order (recommended)

Work bottom-up so each phase can import from the ones before it:

1. `schemas.py` — add `Chunk`, `FigureRecord`, `EvalReport`
2. `ingestion/chunker.py` — depends on `schemas.Chunk`
3. `ingestion/figure_extractor.py` — depends on `schemas.FigureRecord`
4. `retrieval/embedder.py` — depends on nothing external
5. `retrieval/vector_store.py` — depends on `Embedder`, `Chunk`
6. `core/llm.py` — depends on `Settings`
7. `prompts/` — templates + `load_prompt()`
8. `core/rag_pipeline.py` — depends on `VectorStore`, `LLMClient`, `load_prompt`
9. `multimodal/clip_embedder.py`, `multimodal/vlm_client.py` — depends on nothing
10. `evaluation/metrics.py`, `evaluation/eval_pipeline.py` — depends on `VectorStore`, `LLMClient`
11. `observability/logging.py` — depends on `Settings`
12. `core/exceptions.py` — standalone, add first if you want typed error handling from day one

After each step: run `MRTA_ENV=test pytest`, commit.

---

## Current status

| Module | Status |
|--------|--------|
| `core/config.py` | ✅ complete |
| `core/schemas.py` | partial — `PageRecord`, `PdfDocument` done; `Chunk`, `FigureRecord`, `EvalReport` missing |
| `core/llm.py` | stub |
| `core/rag_pipeline.py` | stub |
| `core/exceptions.py` | stub |
| `ingestion/pdf_loader.py` | ✅ complete |
| `ingestion/chunker.py` | stub |
| `ingestion/figure_extractor.py` | stub |
| `retrieval/embedder.py` | stub |
| `retrieval/vector_store.py` | stub |
| `retrieval/reranker.py` | stub |
| `multimodal/clip_embedder.py` | stub |
| `multimodal/vlm_client.py` | stub |
| `prompts/` | stub |
| `evaluation/eval_pipeline.py` | stub |
| `evaluation/metrics.py` | stub |
| `observability/logging.py` | stub |
| `observability/tracing.py` | stub |
