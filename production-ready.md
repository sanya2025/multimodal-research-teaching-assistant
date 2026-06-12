# Production-Ready Plan

## Purpose

This is the **architecture blueprint** for the `src/mrta/` library.
It defines what the library should look like when complete — every module, every public function, every interface signature.

Use it to answer: *"What should I build next, and what should it look like?"*

It does **not** track progress or record what was done in each session — that lives in
[notebook-to-production-steps.md](notebook-to-production-steps.md), the per-notebook execution log.

---

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
│   ├── schemas.py         PageRecord, PdfDocument, Chunk, FigureRecord, EvalReport  ✅ done
│   ├── llm.py             LLMClient — provider-agnostic text generation    ✅ done
│   ├── rag_pipeline.py    rag_query() — retrieve → prompt → generate       ✅ done
│   └── exceptions.py      MrtaError base + subclasses                      ✅ done
├── ingestion/
│   ├── pdf_loader.py      load_pdf(), _doc_id(), ocr_page_if_needed()     ✅ done
│   ├── chunker.py         fixed_chunks(), recursive_chunks(),
│   │                      token_chunks(), semantic_chunks(), chunk_pdf()    ✅ done
│   └── figure_extractor.py extract_figures() → list[FigureRecord]          ✅ done
├── retrieval/
│   ├── embedder.py        Embedder (sentence-transformers + Ollama)        ✅ done
│   ├── vector_store.py    VectorStore (FAISS default, Qdrant swap)         ✅ done
│   └── reranker.py        Reranker (cross-encoder, optional)               stub
├── multimodal/
│   ├── clip_embedder.py   CLIPEmbedder — image → float32 vector            ✅ done
│   └── vlm_client.py      VLMClient — image + text → caption               ✅ done
├── prompts/
│   ├── __init__.py        load_prompt(name, **kwargs) → str                ✅ done
│   ├── rag.j2             base RAG template                                ✅ done
│   ├── explain.j2         explain a figure or concept                      stub
│   ├── quiz.j2            generate quiz questions                          stub
│   ├── beginner.j2        simplified explanation                           stub
│   └── expert.j2          grad-student depth                               stub
├── evaluation/
│   ├── eval_pipeline.py   run_eval(benchmark) → EvalReport                 stub
│   └── metrics.py         answer_relevance(), faithfulness(),              ✅ done
│                          citation_correctness(), hallucination_rate(),
│                          recall_at_k(), mrr(), ndcg_at_k(),
│                          citation_coverage()
└── observability/
    ├── logging.py         StructuredLogger — JSONL per run                 ✅ done
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
| `PageRecord`, `PdfDocument` | `src/mrta/core/schemas.py` | ✅ done — shared across ingestion, retrieval, API |
| `load_pdf(path) -> PdfDocument` | `src/mrta/ingestion/pdf_loader.py` | ✅ done |
| `ocr_page_if_needed(page) -> str` | `src/mrta/ingestion/pdf_loader.py` | ✅ done — OCR fallback for scanned pages |
| `extract_figures(path) -> list[FigureRecord]` | `src/mrta/ingestion/figure_extractor.py` | stub — extracted in Phase 07 |

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
| `Chunk` schema | `src/mrta/core/schemas.py` | ✅ done — used by retrieval, RAG, API, evaluation |
| `fixed_chunks(pdf, size, overlap)` | `src/mrta/ingestion/chunker.py` | ✅ done — baseline strategy |
| `recursive_chunks(pdf, size, overlap)` | `src/mrta/ingestion/chunker.py` | ✅ done — default production strategy |
| `token_chunks(pdf, size, overlap)` | `src/mrta/ingestion/chunker.py` | ✅ done — token-budget-aware |
| `semantic_chunks(pdf, threshold)` | `src/mrta/ingestion/chunker.py` | ✅ done — advanced; slow but highest quality |
| `chunk_pdf(pdf, strategy, **kwargs)` | `src/mrta/ingestion/chunker.py` | ✅ done — single entry point for RAG pipeline |

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

```jinja2
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
| `AskRequest`, `AskResponse` | `apps/api/schemas/ask.py` | ✅ done — Pydantic v2 schemas; versioned independently of the library |
| `UploadResponse` | `apps/api/schemas/upload.py` | ✅ done — Separating schemas by endpoint keeps each small and testable |
| `DocumentInfo` | `apps/api/schemas/documents.py` | ✅ done — Response schema for /documents |
| `/ask` route | `apps/api/routers/ask.py` | ✅ done — Routes import from `mrta.*`; no business logic in routes |
| `/upload` route | `apps/api/routers/upload.py` | ✅ done — Saves PDF to `data/raw/`, calls `load_pdf`, stores in a registry |
| `/documents` route | `apps/api/routers/documents.py` | ✅ done — Lists ingested doc IDs with page and chunk counts |

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

## Phase 06 — Streamlit Frontend ✅ done

**Notebook teaches:** Streamlit layout, file upload widget, calling the FastAPI backend, displaying answers with page citations.

**What to extract:** nothing — all UI code belongs in `apps/streamlit/app.py`. ✅ done

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

## Phase 07 — Figure Extraction & VLM ✅ done

**Notebook teaches:** bounding-box-aware figure extraction, CLIP embeddings for cross-modal retrieval, VLM captioning with LLaVA or Qwen2-VL.

**What to extract:**

| Class / function | Target file | Why | Status |
|-----------------|-------------|-----|--------|
| `extract_figures(path) -> list[FigureRecord]` | `src/mrta/ingestion/figure_extractor.py` | Also needed in Phase 01; single implementation | ✅ done |
| `CLIPEmbedder` | `src/mrta/multimodal/clip_embedder.py` | Image embedding; separate from text embedder | ✅ done |
| `VLMClient` | `src/mrta/multimodal/vlm_client.py` | Wraps Ollama LLaVA and HF Qwen2-VL behind one interface | ✅ done |

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
| `load_prompt(name, **kwargs) -> str` | `src/mrta/prompts/__init__.py` | ✅ done in Phase 04 |
| `MODES` constant | `src/mrta/prompts/__init__.py` | ✅ done — maps mode name → template base name |
| `_base.j2` | `src/mrta/prompts/_base.j2` | ✅ done — shared grounding wrapper; all RAG modes extend it |
| `rag.j2` | `src/mrta/prompts/rag.j2` | ✅ done in Phase 04 |
| `beginner.j2` | `src/mrta/prompts/beginner.j2` | ✅ done — teaching mode |
| `expert.j2` | `src/mrta/prompts/expert.j2` | ✅ done — teaching mode (graduate-level) |
| `quiz.j2` | `src/mrta/prompts/quiz.j2` | ✅ done — teaching mode |
| `explain.j2` | `src/mrta/prompts/explain.j2` | ✅ done — figure explanation for VLMClient |
| `interview.j2` | `src/mrta/prompts/interview.j2` | ✅ done — tutorial extra; system-design interview framing |
| `lecture_notes.j2` | `src/mrta/prompts/lecture_notes.j2` | ✅ done — tutorial extra; structured study notes |

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
| `core/schemas.py` | ✅ complete (`PageRecord`, `PdfDocument`, `Chunk`, `FigureRecord`, `EvalReport`) |
| `core/llm.py` | ✅ complete |
| `core/rag_pipeline.py` | ✅ complete |
| `core/exceptions.py` | ✅ done |
| `ingestion/pdf_loader.py` | ✅ complete |
| `ingestion/chunker.py` | ✅ complete |
| `ingestion/figure_extractor.py` | ✅ complete |
| `retrieval/embedder.py` | ✅ complete |
| `retrieval/vector_store.py` | ✅ complete |
| `retrieval/reranker.py` | ✅ complete (`Reranker` wrapping `CrossEncoder`; integrated into `rag_query`) |
| `multimodal/clip_embedder.py` | ✅ complete |
| `multimodal/vlm_client.py` | ✅ complete |
| `prompts/` | ✅ complete (all templates done: rag, _base, beginner, expert, quiz, explain, interview, lecture_notes) |
| `evaluation/eval_pipeline.py` | ✅ complete (`run_eval` returning `EvalReport`) |
| `evaluation/metrics.py` | ✅ complete (`answer_relevance`, `faithfulness`, `citation_correctness`, `hallucination_rate`, `recall_at_k`, `mrr`, `ndcg_at_k`, `citation_coverage`) |
| `observability/logging.py` | ✅ complete |

---

## feature/exception-hierarchy

### Understanding

**Current implementation:**

- `src/mrta/core/exceptions.py` is a stub — one docstring, nothing else.
- One explicit bare raise in the library: `chunker.py:158` → `ValueError`.
- Five implicit failure points where third-party errors currently escape unwrapped: `fitz.open`
  (IngestionError), `httpx.raise_for_status` in `embedder.py` (EmbeddingError),
  `faiss.read_index` in `vector_store.py` (RetrievalError), `ollama.chat` in `llm.py`
  and `vlm_client.py` (LLMError).
- The API upload router already raises `HTTPException` for non-PDF — correct, that's a
  boundary concern, not a library concern.

**Relevant files:**

- `src/mrta/core/exceptions.py` — fill in
- `src/mrta/ingestion/chunker.py` — replace `ValueError`
- `src/mrta/ingestion/pdf_loader.py` — wrap `fitz.open`
- `src/mrta/retrieval/embedder.py` — wrap `httpx.raise_for_status`
- `src/mrta/retrieval/vector_store.py` — wrap `faiss.read_index`
- `src/mrta/core/llm.py` — wrap `ollama.chat`
- `src/mrta/multimodal/vlm_client.py` — wrap `ollama.chat`
- `src/mrta/__init__.py` — export all exception classes
- `tests/unit/test_exceptions.py` — new, covers hierarchy + each raise site

**Risks:**

- Wrapping too broadly hides the original traceback. Mitigation: always use
  `raise XError("...") from e` to preserve `__cause__`.
- Changing `ValueError` to `IngestionError` is a breaking change for any caller catching
  `ValueError`. Acceptable pre-1.0 with no external callers.
- `ocr_page_if_needed` has an intentional silent fallback — leave it alone.
- `clip_embedder.py` constructor errors come from optional deps not being installed;
  wrapping them as `EmbeddingError` would obscure the "install [multimodal]" message.
  Leave it unwrapped.

### Hierarchy

```python
MRTAError(Exception)       # base
├── IngestionError         # PDF load, chunking, figure extraction
├── EmbeddingError         # sentence-transformers, Ollama embed, CLIP
├── RetrievalError         # FAISS operations (load, search)
├── LLMError               # ollama.chat in LLMClient + VLMClient
└── EvaluationError        # eval_pipeline, metrics (future-proofing)
```

No extra fields — just subclasses. Message + `raise ... from e` pattern carries all context.

### Steps

**1 — Fill in `exceptions.py`** — define `MRTAError` + five subclasses, all exported.

**2 — `chunker.py:158`** — replace `ValueError` with `IngestionError`.

**3 — `pdf_loader.py:load_pdf`** — wrap `fitz.open`:

```python
try:
    doc = fitz.open(path)
except Exception as e:
    raise IngestionError(f"Cannot open PDF {path}: {e}") from e
```

**4 — `embedder.py:_embed_ollama`** — wrap `raise_for_status`:

```python
try:
    resp.raise_for_status()
except httpx.HTTPStatusError as e:
    raise EmbeddingError(f"Ollama embed request failed ({e.response.status_code}): {e}") from e
```

**5 — `vector_store.py:load`** — wrap `faiss.read_index`:

```python
try:
    store._index = faiss.read_index(str(p / "index.faiss"))
except Exception as e:
    raise RetrievalError(f"Cannot load FAISS index from {p}: {e}") from e
```

**6 — `llm.py` and `vlm_client.py`** — wrap `ollama.chat`:

```python
try:
    resp = ollama.chat(...)
except Exception as e:
    raise LLMError(f"Ollama chat failed (model={self._model}): {e}") from e
```

**7 — `src/mrta/__init__.py`** — add all five concrete classes + base to `__all__`.

**8 — `tests/unit/test_exceptions.py`** — 9 tests:

- Hierarchy: each subclass `isinstance` of `MRTAError`; `MRTAError` is an `Exception`
- One test per wrapped raise site using mocks; assert correct subclass raised and `__cause__` set
- Update the existing `chunker` test that asserts `ValueError` → assert `IngestionError`

### Expected outcome

- `from mrta import IngestionError, LLMError` etc. works.
- All third-party errors at system boundaries re-raised with a typed mrta exception and preserved `__cause__`.
- 9 new tests; all 107 existing tests continue to pass.

---

## feature/upload-validation

### Understanding

**Current implementation:**

- `apps/api/routers/upload.py` — single check: `file.filename.lower().endswith(".pdf")`.
  Extension check only; no size limit, no MIME type check, no PDF magic-byte check, no
  filename sanitisation, no structured error body.
- `file.filename` is written directly to `data/raw/` without sanitisation — path-traversal
  risk if a client sends `../../etc/passwd.pdf`.
- The entire file is read into memory with `await file.read()` before any size check —
  a large upload exhausts server memory before it can be rejected.
- `load_pdf` receives the saved file; a malformed PDF causes `fitz` to raise a bare
  `Exception` that propagates as a 500 Internal Server Error.
- `apps/api/schemas/upload.py` — `UploadResponse` only. No error schema; FastAPI returns
  its default `{"detail": "..."}` shape for 4xx.
- `tests/unit/test_api.py` — `TestUpload` has 3 tests: 200 happy path (×2) + 400 for
  non-PDF extension. No tests for size, MIME, path traversal, or malformed PDF.

**Relevant files:**

- `apps/api/routers/upload.py` — all validation logic lives here
- `apps/api/schemas/upload.py` — add `UploadError` response schema
- `apps/api/main.py` — add `MAX_UPLOAD_BYTES` constant; wire exception handler
- `tests/unit/test_api.py` — extend `TestUpload` with new cases

**Dependencies:**

- `python-magic` (or stdlib `imghdr` alternative) for MIME sniffing — **avoid**: adds a
  system-level `libmagic` dependency. Use the PDF magic bytes directly instead:
  first 4 bytes of a valid PDF are `%PDF`.
- No new library dependencies needed.

**Risks:**

- Reading `await file.read()` a second time returns empty bytes — must read once, store,
  then reuse the bytes for both size check and magic-byte check before writing.
- `Path(file.filename).name` strips directory components but not all OS-specific separators
  on Windows paths. Use `Path(file.filename).name` and then validate no `..` remains.
- Changing the 400 response body shape (adding `UploadError`) is technically a breaking
  change for any client parsing `{"detail": "..."}` — acceptable pre-1.0 if Streamlit
  `app.py` is updated at the same time.
- `IngestionError` from `load_pdf` (added in `feat/exception-hierarchy`) should map to
  422 Unprocessable Entity, not 500 — add an exception handler in `main.py`.

### Design

**Validation order** (fail-fast, cheapest first):

```text
1. Filename extension    → 400  "Only PDF files are accepted."
2. File size             → 413  "File exceeds 20 MB limit."
3. PDF magic bytes       → 415  "File does not appear to be a valid PDF."
4. Safe filename         → sanitise silently (strip path components, slugify)
5. load_pdf (fitz)       → 422  "PDF is malformed or unreadable." (via IngestionError)
```

**`UploadError` schema** (add to `apps/api/schemas/upload.py`):

```python
class UploadError(BaseModel):
    detail: str
    code: str   # "invalid_extension" | "file_too_large" | "invalid_mime" | "malformed_pdf"
```

**Constants** (add to `apps/api/routers/upload.py`):

```python
MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB
PDF_MAGIC = b"%PDF"
```

### Steps

**1 — `apps/api/schemas/upload.py`** — add `UploadError(BaseModel)` with `detail: str`
and `code: str`.

**2 — `apps/api/routers/upload.py`** — rewrite the upload handler:

```python
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PDF_MAGIC = b"%PDF"

@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), store=Depends(get_store)) -> UploadResponse:
    # 1. extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    # 2. read once
    data = await file.read()
    # 3. size
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")
    # 4. magic bytes
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(status_code=415, detail="File does not appear to be a valid PDF.")
    # 5. safe filename
    safe_name = Path(file.filename).name
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / safe_name
    path.write_bytes(data)
    # 6. parse — IngestionError caught by global handler → 422
    pdf = load_pdf(path)
    chunks = chunk_pdf(pdf, strategy="recursive")
    store.add(chunks)
    store.save(settings.vector_store_path / "default")
    return UploadResponse(
        doc_id=pdf.doc_id,
        source=pdf.source,
        n_pages=pdf.n_pages,
        n_chunks=len(chunks),
    )
```

**3 — `apps/api/main.py`** — add `IngestionError` exception handler:

```python
from mrta.core.exceptions import IngestionError

@app.exception_handler(IngestionError)
async def ingestion_error_handler(request, exc: IngestionError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "code": "malformed_pdf"})
```

**4 — `tests/unit/test_api.py`** — extend `TestUpload`:

```python
def test_oversized_file_returns_413(self, client): ...
def test_non_pdf_magic_bytes_returns_415(self, client): ...
def test_path_traversal_filename_is_sanitised(self, client): ...
def test_malformed_pdf_returns_422(self, client): ...
```

### Expected outcome

- Five validation layers in place; each reachable via a dedicated test.
- No path-traversal risk: `../../evil.pdf` → saved as `evil.pdf`.
- Oversized uploads rejected before disk write.
- Corrupt PDFs return 422, not 500.
- 4 new tests added to `TestUpload` (total 7); all existing tests continue to pass.

---

## chore/ci-quality-gates

### Understanding

**Current CI (`ci.yml`) — what exists:**

- One job (`test`) with 5 steps: checkout, Python 3.11 setup, `pip install -e ".[dev,api]"`,
  ruff lint, black format check, pytest.
- No type checking (mypy/pyright).
- No dependency vulnerability scanning (pip-audit).
- No FastAPI smoke test (health endpoint is never hit in CI).
- No Docker build check (Dockerfile.api could silently break).

**Relevant files:**

- `.github/workflows/ci.yml` — the only CI file; all changes go here
- `docker/Dockerfile.api` — used by the Docker build check step
- `pyproject.toml` — add `mypy` and `pip-audit` to the `dev` optional group

**Risks:**

- `mypy` on a mixed codebase often surfaces pre-existing annotation gaps. Run with
  `--ignore-missing-imports` and `--no-strict-optional` initially to avoid blocking CI on
  third-party stubs.
- `pip-audit` exits non-zero when vulnerabilities are found. Pin a `--ignore-vuln` only
  if a CVE has no fix yet and is documented.
- Docker build in CI requires no secrets — `Dockerfile.api` uses only `pip install` and
  `apt-get`, so it is safe to build without `.env`.
- FastAPI smoke test must not depend on Ollama or FAISS being live. Use
  `MRTA_ENV=test uvicorn ... &` + `curl /health` — the lifespan patches in test env
  already skip model loading.
- Adding new CI steps increases total run time. Each new step is estimated: mypy ~30s,
  pip-audit ~20s, Docker build ~90s, smoke test ~15s. Total addition: ~2.5 min.

### Design

**New CI job layout** — keep the existing `test` job unchanged; add three new jobs that
run in parallel after `test` passes:

```text
test  (existing — lint + format + pytest)
  ├── type-check   (mypy on src/ apps/)
  ├── audit        (pip-audit)
  └── docker       (docker build + smoke test)
```

Each new job uses `needs: test` so they only run on a green test suite.

**mypy configuration** — add to `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
exclude = ["tests/", "notebooks/"]
```

**pip-audit** — run as a standalone step; no configuration file needed.

**Docker smoke test** — build `Dockerfile.api`, start container with `MRTA_ENV=test`,
hit `GET /health`, assert `{"status": "ok"}`, stop container.

### Steps

**1 — `pyproject.toml`** — add `mypy` and `pip-audit` to the `dev` group:

```toml
dev = [
    ...
    "mypy>=1.10.0",
    "pip-audit>=2.7.0",
]
```

Also add `[tool.mypy]` section (see Design above).

**2 — `.github/workflows/ci.yml`** — add three jobs after the existing `test` job:

```yaml
  type-check:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4.2.2
      - uses: actions/setup-python@v5.4.0
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -e ".[dev,api]"
      - name: Type check (mypy)
        run: mypy src/ apps/ --ignore-missing-imports

  audit:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4.2.2
      - uses: actions/setup-python@v5.4.0
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -e ".[dev,api]"
      - name: Vulnerability scan (pip-audit)
        # --skip-editable: mrta is a local editable install, not on PyPI
        # --ignore-vuln CVE-2025-3000: torch vulnerability with no fix available yet
        run: pip install --upgrade pip && pip install -e ".[dev,api]"
        run: pip-audit --skip-editable --ignore-vuln CVE-2025-3000

  docker:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4.2.2
      - name: Build API image
        run: docker build -f docker/Dockerfile.api -t mrta-api:ci .
      - name: Smoke test /health
        run: |
          docker run -d --name mrta-ci -e MRTA_ENV=test -p 8000:8000 mrta-api:ci
          for i in 1 2 3 4 5; do
            sleep 6
            curl --fail --silent http://localhost:8000/health && docker stop mrta-ci && exit 0
            echo "Attempt $i failed, retrying..."
          done
          echo "=== Container logs ==="
          docker logs mrta-ci
          docker stop mrta-ci
          exit 1
```

### Actual outcome (shipped)

- CI has 4 jobs: `test`, `type-check`, `audit`, `docker`. ✅
- `type-check` surfaced 6 pre-existing mypy annotation gaps in 4 source files — all fixed as part of this branch:
  - `src/mrta/core/config.py` — `settings_customise_sources` signature
  - `src/mrta/retrieval/embedder.py` — `SentenceTransformer | None` annotation + `TYPE_CHECKING` guard
  - `src/mrta/retrieval/vector_store.py` — `_index: faiss.Index` annotation
  - `apps/api/routers/upload.py` — `str | None` guard on `file.filename`
- `audit` required `--skip-editable` (mrta not on PyPI) and `--ignore-vuln CVE-2025-3000` (torch, no fix available). pip upgraded to 26.1.2 to clear PYSEC-2026-196.
- `docker` required `COPY LICENSE ./` and `COPY README.md ./` in `Dockerfile.api` — hatchling requires both at build time. Smoke test uses a 5-attempt retry loop with container log dump on failure.
- No new test files.
- All 119 existing tests continue to pass.

---

## chore/docker-healthchecks

### Understanding

**Current implementation:**

- `docker-compose.yml` uses bare `depends_on` — waits for a container to *start*, not for
  the service inside to be *ready*. Ollama can take 10–30 s to load a model; the API starts
  immediately and fails its first embed/chat call if Ollama is not yet up.
- No `HEALTHCHECK` instruction in `Dockerfile.api` or `Dockerfile.streamlit`.
- No `healthcheck` blocks in `docker-compose.yml`.
- Result: `docker compose up` in local dev is unreliable — API often starts before Ollama,
  and Streamlit sometimes starts before the API `/health` route is live.

**Relevant files:**

- `docker/docker-compose.yml` — add healthcheck blocks; upgrade depends_on to condition-based
- `docker/Dockerfile.api` — add HEALTHCHECK instruction
- `docker/Dockerfile.streamlit` — add HEALTHCHECK instruction

**Dependencies:**

- `python:3.11-slim` does not include `curl`. Use a Python one-liner for HEALTHCHECK to
  avoid adding a new system package:
  `python -c "import urllib.request; urllib.request.urlopen('http://localhost:PORT/PATH')"`
- Ollama image (`ollama/ollama:latest`) does include curl — use it for the Ollama healthcheck.
- Streamlit's built-in health endpoint is `/_stcore/health`, not `/health`.
- Ollama's health endpoint is `GET /api/tags` (returns model list; 200 when ready).

**Risks:**

- `start-period` must be generous for Ollama (model loading can exceed 30 s on first pull).
  Use `--start-period=60s` for Ollama, `--start-period=30s` for API and Streamlit.
- `condition: service_healthy` requires Compose v2 — already satisfied (no `version:` key
  in the file; Docker Compose v2 is the default since Docker Desktop 4.x).
- Do not set `interval` too short — 15 s is safe for local dev; CI uses its own smoke test.

### Design

```text
ollama   (healthcheck: GET /api/tags → 200)
  └── api        (depends_on: ollama condition: service_healthy)
                 (healthcheck: GET /health → 200)
        └── streamlit  (depends_on: api condition: service_healthy)
                       (healthcheck: GET /_stcore/health → 200)
```

HEALTHCHECK timing per service:

| Service | interval | timeout | start-period | retries |
|---------|----------|---------|--------------|---------|
| ollama (compose only) | 15s | 5s | 60s | 5 |
| api (Dockerfile + compose) | 15s | 5s | 30s | 3 |
| streamlit (Dockerfile + compose) | 15s | 5s | 30s | 3 |

### Steps

**1 — `docker/Dockerfile.api`** — add HEALTHCHECK after EXPOSE:

```dockerfile
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

**2 — `docker/Dockerfile.streamlit`** — add HEALTHCHECK after EXPOSE:

```dockerfile
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
```

**3 — `docker/docker-compose.yml`** — add `healthcheck` block to each service and upgrade
`depends_on` to condition-based.

### Expected outcome

- `docker compose up` starts services in dependency order and waits for each to pass its
  healthcheck before starting the next.
- `docker compose ps` shows `(healthy)` for all three services once fully up.
- API never attempts an Ollama call before Ollama is ready.
- No new tests — this is pure Docker configuration.

---

## feat/cross-encoder-reranking

### Understanding

**Current implementation:**

- `src/mrta/retrieval/` had `embedder.py` and `vector_store.py` — no `reranker.py`.
- `src/mrta/retrieval/__init__.py` exported only `Embedder` and `VectorStore`.
- `src/mrta/core/rag_pipeline.py` — `rag_query()` called `vector_store.search(question, k=top_k)`
  and passed results directly to the prompt. No reranking step existed.

**Relevant files:**

- `src/mrta/retrieval/reranker.py` — created: `Reranker` class wrapping `CrossEncoder`
- `src/mrta/retrieval/__init__.py` — updated: exports `Reranker`
- `src/mrta/core/rag_pipeline.py` — updated: `rag_query()` gains optional `reranker` / `rerank_top_n`
- `tests/unit/test_reranker.py` — created: 8 unit tests

**Note on dependencies:**

`sentence-transformers>=2.7.0` is already in core `dependencies` in `pyproject.toml` — no new
optional extra was needed.

**Risks addressed:**

- `rag_query()` signature change is fully backwards-compatible — `reranker=None` default keeps
  all existing callers working unchanged.
- `CrossEncoder` is mocked in tests via `patch("sentence_transformers.CrossEncoder")` — no
  model download in CI.
- `rerank_top_n` > `len(chunks)` is handled silently: `ranked[:top_n]` returns all chunks.

### Design

```text
rag_query(question, vector_store, llm, top_k=5, reranker=None, rerank_top_n=3)
  │
  ├── vector_store.search(question, k=top_k)   → list[Chunk]  (broad recall)
  │
  ├── [optional] reranker.rerank(question, chunks, top_n=rerank_top_n)
  │                                             → list[Chunk]  (precision-sorted)
  │
  └── load_prompt("rag", chunks=sources, question=question) → LLM → answer
```

`Reranker` interface:

```python
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None: ...
    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 3) -> list[Chunk]: ...
```

### Expected outcome

- `from mrta.retrieval import Reranker` works.
- `rag_query(..., reranker=None)` behaves identically to the original implementation.
- `rag_query(..., reranker=reranker, rerank_top_n=3)` calls `reranker.rerank()` and passes
  reranked chunks to the LLM instead of raw vector-search results.
- All existing `test_rag_pipeline.py` tests pass unchanged.
- 8 new tests in `tests/unit/test_reranker.py` — all passing.
