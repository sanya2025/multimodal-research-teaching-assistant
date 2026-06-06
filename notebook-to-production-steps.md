# Notebook-to-Production Steps

## Purpose

This is the **execution log** for converting tutorial notebooks into production library code.
It records — notebook by notebook — which cells run, which functions were extracted, and what
the production notebook looks like after each phase.

Use it to answer: *"What has already been done, and where does the next session pick up?"*

It does **not** define what the library should look like when complete — that lives in
[production-ready.md](production-ready.md), the architecture blueprint.

---

Per-notebook guide: what to run, what to extract, and what the production version looks like.

---

## Phase 00 — Foundations & Setup

Notebook 00 is purely a setup verification notebook — no code to write. Here's what each section does when you run it:

| Cell | What it does | Requires |
|------|--------------|----------|
| [0]  | Stale TODO — already done (package exists) | Delete or ignore |
| [4]  | Prints Python version and platform | Nothing |
| [6]  | Checks Ollama is running at localhost:11434 | `ollama serve` running |
| [8]  | First LLM call — `llama3.2:latest` | `ollama pull llama3.2:3b` |
| [11] | Downloads MiniLM embedder from HF | Internet (first run only) |
| [12] | Embedding call — `nomic-embed-text` | `ollama pull nomic-embed-text` |
| [14] | Checks all key imports work | `pip install -e ".[all]"` |
| [16] | Loads `mrta.settings` and prints config values | Nothing (already installed) |

**So: run it top to bottom.** If all cells pass without error, your environment is ready for Phase 01 onward.

The one thing to clean up first: delete cell [0] (the TODO). It was the pre-Phase-1 reminder to build the package — that's done. The production notebook already has it removed; the tutorial notebook still has it as a stale artifact.

---

## Phase 01 — PDF Ingestion

Phase 01 is the first notebook with real library code. Two functions have already been extracted to
`src/mrta/` and are fully working — the production notebook imports them directly.

### What's extracted (production imports active)

| Tutorial cell | Inline code | Production import |
|---------------|-------------|-------------------|
| [6]  | `class PageRecord(BaseModel): ...` `class PdfDocument(BaseModel): ...` | `from mrta.core.schemas import PageRecord, PdfDocument` |
| [8]  | `def _doc_id(path): ...` `def load_pdf(path): ...` | `from mrta.ingestion import load_pdf` |
| [17] | `def ocr_page_if_needed(page, dpi): ...` | `from mrta.ingestion.pdf_loader import ocr_page_if_needed` |

### What's still inline (extracted in a later phase)

| Tutorial cell | Inline code | Extracted when |
|---------------|-------------|----------------|
| [12–14] | Figure saving loop + PIL display | Phase 07 — becomes `src/mrta/ingestion/figure_extractor.py` |

### Running the production notebook

The production notebook (`notebooks/production/`) is **fully runnable as-is**.

| Cell | What it does | Status |
|------|--------------|--------|
| [3]  | Loads PDF from `data/sample/` — no download | ✅ runs |
| [5]  | Opens PDF with PyMuPDF, prints metadata | ✅ runs |
| [7]  | Imports `PageRecord`, `PdfDocument` from `mrta.core.schemas` | ✅ runs |
| [9]  | Imports and calls `load_pdf()` from `mrta.ingestion` | ✅ runs |
| [11] | Builds a pandas DataFrame of pages | ✅ runs |
| [13] | Extracts and saves embedded figures to `data/processed/figures/` | ✅ runs (inline) |
| [15] | Displays one figure inline | ✅ runs |
| [18] | Imports `ocr_page_if_needed` from `mrta.ingestion.pdf_loader` | ✅ runs |

**Requires:** `pip install -e ".[all]"` and `data/sample/attention_is_all_you_need.pdf` (already git-tracked).

### Functions implemented in Phase 01

All three live in `src/mrta/ingestion/pdf_loader.py`. The schemas live in `src/mrta/core/schemas.py`.

| Function / class | Signature | What it does |
|-----------------|-----------|--------------|
| `_doc_id` | `(path: Path) -> str` | SHA-1 hash of file bytes → stable `"{stem}_{hash[:10]}"` ID |
| `load_pdf` | `(path: str \| Path) -> PdfDocument` | Opens PDF, extracts text + image count per page, returns typed `PdfDocument` |
| `ocr_page_if_needed` | `(page: fitz.Page, dpi: int = 200) -> str` | Returns embedded text if present; falls back to pytesseract OCR for scanned pages |
| `PageRecord` | Pydantic model | One page: `doc_id`, `page`, `text`, `n_images`, `source` |
| `PdfDocument` | Pydantic model | Full document: `doc_id`, `source`, `title`, `n_pages`, `pages: list[PageRecord]` |

### Concept: SHA-1 as a stable document ID

SHA-1 is a hashing algorithm — feed it any bytes, get a fixed 40-character hex string back.
The same input always produces the same output; any change to the content produces a
completely different output.

`_doc_id` hashes the raw PDF bytes so the ID is tied to the file content, not the filename:

```python
def _doc_id(path: Path) -> str:
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem}_{h}"
```

| File | Content changed? | `doc_id` |
|------|-----------------|----------|
| `attention_is_all_you_need.pdf` | — | `attention_is_all_you_need_a3f2c91b` |
| `attention_is_all_you_need.pdf` | no (same file) | `attention_is_all_you_need_a3f2c91b` (identical) |
| `paper_renamed.pdf` | no (just renamed) | `paper_renamed_a3f2c91b` (same hash suffix) |
| `attention_is_all_you_need_v2.pdf` | yes | `attention_is_all_you_need_v2_7d4e110f` (different) |

**Why this matters for RAG:** every `PageRecord` stores `doc_id`. When the LLM cites page 7,
the system resolves `doc_id` back to the source file. If the ID changed between runs,
all citations would break. Hashing the content guarantees stability as long as the file
doesn't change.

---

## Phase 02 — Chunking Strategies

Phase 02 introduces the `Chunk` schema and four chunking strategies. All five functions
plus the dispatcher have been extracted to `src/mrta/ingestion/chunker.py`.

### What's extracted — Phase 02 (production imports active)

| Tutorial cell | Inline code | Production import |
|---------------|-------------|-------------------|
| [4] | `class PageRecord/PdfDocument`, `load_pdf` (re-defined) | `from mrta.core.schemas import PageRecord, PdfDocument` + `from mrta.ingestion import load_pdf` |
| [6] | `class Chunk(BaseModel): ...` | `from mrta.core.schemas import Chunk` |
| [8] | `def fixed_chunks(pdf, size, overlap): ...` | `from mrta.ingestion.chunker import fixed_chunks` |
| [10] | `def recursive_chunks(pdf, size, overlap): ...` | `from mrta.ingestion.chunker import recursive_chunks` |
| [12] | `def token_chunks(pdf, size, overlap): ...` | `from mrta.ingestion.chunker import token_chunks` |
| [15] | `def semantic_chunks(pdf, threshold, max_chars): ...` | `from mrta.ingestion.chunker import semantic_chunks` |

### What's still inline (stays in notebook)

| Tutorial cell | Inline code | Why it stays |
|---------------|-------------|--------------|
| [14] | `def summarize(name, chunks): ...` | Notebook display helper — not reusable library code |

### Running the Phase 02 production notebook

| Cell | What it does | Status |
|------|--------------|--------|
| [4] | Imports `PageRecord`, `PdfDocument`, `load_pdf`; loads from `data/sample/` | ✅ runs |
| [6] | Imports `Chunk` from `mrta.core.schemas` | ✅ runs |
| [8] | Imports + calls `fixed_chunks()` | ✅ runs |
| [10] | Imports + calls `recursive_chunks()` | ✅ runs |
| [12] | Imports + calls `token_chunks()` | ✅ runs |
| [14] | Inline `summarize()` + comparison DataFrame | ✅ runs (inline) |
| [17] | Imports + calls `semantic_chunks()` on first 5 pages | ✅ runs |
| [19] | Saves `tokenized` chunks to `data/processed/chunks.jsonl` | ✅ runs |

**Requires:** `pip install -e ".[all]"` and `data/sample/attention_is_all_you_need.pdf`.

### Functions implemented in Phase 02

Functions live in `src/mrta/ingestion/chunker.py`. `Chunk` lives in `src/mrta/core/schemas.py`.

| Function / class | Signature | What it does |
|-----------------|-----------|--------------|
| `Chunk` | Pydantic model | One chunk: `chunk_id`, `doc_id`, `source`, `page`, `text`, `section?`, `n_tokens?` |
| `fixed_chunks` | `(pdf, size=1500, overlap=200) -> list[Chunk]` | Naive fixed-width character windows |
| `recursive_chunks` | `(pdf, size=800, overlap=100) -> list[Chunk]` | LangChain splitter — respects paragraph/sentence boundaries |
| `token_chunks` | `(pdf, size=700, overlap=100) -> list[Chunk]` | tiktoken-based — guarantees chunks fit a token budget |
| `semantic_chunks` | `(pdf, threshold=0.55, max_chars=1200) -> list[Chunk]` | Merges sentences by embedding similarity — slow, highest quality |
| `chunk_pdf` | `(pdf, strategy="recursive", **kwargs) -> list[Chunk]` | Dispatcher — routes to the right strategy by name |

### Concept: chunk_id encodes provenance

Every chunk carries a `chunk_id` of the form `{doc_id}_p{page}_c{idx}`:

```text
attention_is_all_you_need_a3f2c91b_p3_c2
│                            │      │  │
└─ stem                      └─hash │  └─ chunk index on this page
                                    └─ page number (1-indexed)
```

This means the RAG system can resolve any citation back to the exact page and chunk without
keeping a separate lookup table — the location is encoded in the ID itself.

---

## Phase 03 — Embeddings & FAISS

Phase 03 introduces `Embedder` (new wrapper class) and `VectorStore` (extracted from the
tutorial and refined). Both live in `src/mrta/retrieval/`.

### What's extracted — Phase 03 (production imports active)

| Tutorial cell | Inline code | Production import |
|---------------|-------------|-------------------|
| [4] | `class Chunk(BaseModel): ...` (re-defined inline) | `from mrta.core.schemas import Chunk` |
| [6] | `embedder = SentenceTransformer(EMBEDDER_NAME)` | `from mrta.retrieval.embedder import Embedder` + `embedder = Embedder()` |
| [14] | `class VectorStore: ...` (full definition) | `from mrta.retrieval.vector_store import VectorStore` |

### What's still inline — Phase 03

| Tutorial cell | Inline code | Why it stays |
|---------------|-------------|--------------|
| [8] | Raw `embedder.encode(...)` with timing | Teaches the embedding API directly |
| [10] | Raw `faiss.IndexFlatIP` build + `index.add` | Teaches the FAISS lifecycle step-by-step |
| [12] | Raw `index.search` query | Teaches the raw query path before abstraction |
| [16] | HNSW index demo (commented out) | Scale reference — not production default |
| [18] | Sanity-check retrieval loop | Teaching quality check, not reusable logic |

### Running the Phase 03 production notebook

| Cell | What it does | Status |
|------|--------------|--------|
| [4] | Imports `Chunk` from `mrta.core.schemas`; loads from `data/processed/chunks.jsonl` | ✅ runs |
| [6] | Imports `Embedder`; instantiates with `settings.embedding_model`; prints `dim` | ✅ runs |
| [8] | Raw encode — teaches the embedding API | ✅ runs (inline) |
| [10] | Raw FAISS index build | ✅ runs (inline) |
| [12] | Raw query against index | ✅ runs (inline) |
| [14] | Imports `VectorStore`; `store.add` / `store.save` / `reloaded.search` demo | ✅ runs |
| [16] | HNSW demo (commented) | ✅ runs (inline) |
| [18] | Sanity checks — returns `list[Chunk]` now (`.page` not `["page"]`) | ✅ runs |

**Requires:** `pip install -e ".[all]"`, `data/processed/chunks.jsonl` (produced by Phase 02).

### Classes implemented in Phase 03

`Embedder` lives in `src/mrta/retrieval/embedder.py`.
`VectorStore` lives in `src/mrta/retrieval/vector_store.py`.

| Class / method | Signature | What it does |
|---------------|-----------|--------------|
| `Embedder.__init__` | `(model_name: str \| None = None) -> None` | Reads `settings.embedding_model`; `model_name` overrides |
| `Embedder.embed` | `(texts: list[str]) -> np.ndarray` | Returns float32, L2-normalised, shape `(n, dim)` |
| `Embedder.dim` | property → `int` | Embedding dimensionality (lazy-loaded) |
| `Embedder.model_name` | property → `str` | The model identifier |
| `VectorStore.__init__` | `(embedder: Embedder) -> None` | Creates `IndexFlatIP(embedder.dim)` |
| `VectorStore.add` | `(chunks: list[Chunk]) -> None` | Embeds and appends to the index |
| `VectorStore.search` | `(query: str, k: int = 5) -> list[Chunk]` | Returns top-k Chunks by cosine similarity |
| `VectorStore.save` | `(path: Path) -> None` | Writes `index.faiss` + `metadata.jsonl` + `config.json` |
| `VectorStore.load` | `(path: Path, embedder: Embedder) -> VectorStore` | Reloads a persisted store |

### Concept: why normalize + IndexFlatIP = cosine similarity

Cosine similarity between two vectors is their dot product divided by the product of their
norms. If both vectors are already L2-normalized (norm = 1), that division is always `1 × 1 = 1`,
so **cosine similarity collapses to a plain dot product**.

`IndexFlatIP` computes exact dot products (inner products) — so feeding it normalized vectors
gives exact cosine similarity search with no training, no approximation, and instant build time.

```text
cos(u, v) = (u · v) / (‖u‖ · ‖v‖)

When ‖u‖ = ‖v‖ = 1:
cos(u, v) = u · v     ← that's exactly what IndexFlatIP computes
```

At paper scale (<1M vectors), `IndexFlatIP` is the right default: exact, instant, no
configuration. The `VectorStore` interface is the swap boundary for Qdrant (Phase 09) —
replacing the implementation doesn't change any caller.

---

## Phase 04 — End-to-End RAG Pipeline

### What's extracted — Phase 04

| Tutorial cell | Inline code | Production import |
|---------------|-------------|-------------------|
| [6] | `class OllamaLLM:` with `chat(system, user) -> dict` | `from mrta.core.llm import LLMClient` — upgraded interface: `chat(messages: list[dict]) -> str` |
| [4] | `RAG_PROMPT = Template(r"""...""")` | `from mrta.prompts import load_prompt` — template lives at `src/mrta/prompts/rag.j2` |
| [8] | `def rag_answer(question, store, llm, k)` | `from mrta.core.rag_pipeline import rag_query` — upgraded return: `{"answer", "sources", "latency_s"}` |
| [14] | `def log_run(run, log_file)` | `from mrta.observability.logging import StructuredLogger` — reads `settings.log_file` |

### What's still inline — Phase 04

| Cell | Content | Why |
|------|---------|-----|
| [12] | Single question demo call | Live Ollama call; teaching demo |
| [14] | Multi-question batch loop | Live Ollama calls; demonstrates batch usage |
| [17] | Failure modes table | Markdown only |

### Running notebook cell status

| Cell | Status | Notes |
|------|--------|-------|
| [4] | `from mrta.retrieval import Embedder, VectorStore` | Phase 03 done — no more inline VectorStore |
| [6] | `from mrta.prompts import load_prompt` | Template body moved to `rag.j2` |
| [8] | `from mrta.core.llm import LLMClient` | Replaces inline `OllamaLLM` |
| [10] | `from mrta.core.rag_pipeline import rag_query` | Replaces inline `rag_answer` |
| [12] | Inline — updated | Uses `rag_query`; accesses `out["sources"]` (list[Chunk], `.page`) |
| [14] | Inline — updated | Loop uses `rag_query`; `[c.page for c in out["sources"]]` |
| [16] | `from mrta.observability.logging import StructuredLogger` | Replaces inline `log_run` |

### Classes and functions implemented — Phase 04

| Symbol | File | Signature |
|--------|------|-----------|
| `LLMClient.__init__` | `src/mrta/core/llm.py` | `(provider: str \| None, model: str \| None) -> None` |
| `LLMClient.chat` | `src/mrta/core/llm.py` | `(messages: list[dict], temperature: float = 0.1) -> str` |
| `rag_query` | `src/mrta/core/rag_pipeline.py` | `(question, vector_store, llm, top_k=5) -> dict` |
| `load_prompt` | `src/mrta/prompts/__init__.py` | `(name: str, **kwargs) -> str` |
| `StructuredLogger.log_run` | `src/mrta/observability/logging.py` | `(question, answer, sources, latency_s) -> None` |

### Two interface refinements from tutorial → production

**`OllamaLLM.chat(system, user)` → `LLMClient.chat(messages: list[dict]) -> str`**

The tutorial API is Ollama-specific: `chat(system, user)` bakes in the two-role convention at the
call site, and returns a dict with `text`, `model`, `latency_s`, `prompt_tokens`. The production
interface uses the OpenAI-style `messages` list — provider-agnostic; any provider that accepts
`[{"role": "system"}, {"role": "user"}]` works without changes to callers. Return type is plain
`str` (response content only); latency and token counts belong at the pipeline level, not inside
the LLM wrapper.

**`rag_answer` return → `rag_query` return**

Tutorial: `{"question", "answer", "cited_pages", "retrieved": [{"page", "score", "chunk_id"}], "latency_s", "model"}`.
Production: `{"answer": str, "sources": list[Chunk], "latency_s": float}`.
The `sources` are typed `Chunk` objects — callers use `.page`, `.text`, `.source` directly instead
of indexing dicts. Keeping the model name out of the return value is intentional: it's an
implementation detail of `LLMClient`, not a property of the answer.

### Concept note — why the RAG prompt has three separate sections

The `rag.j2` template separates the prompt into three distinct regions:

```text
1. Role + grounding rule  (system layer)
2. --- CONTEXT ---        (retrieved chunks, each labeled with source + page)
3. --- QUESTION ---       (user query, placed after context)
```

**Grounding rule prevents hallucination.** "Use ONLY the context below. If insufficient, say
'I could not find this.'" gives the model explicit permission to refuse rather than invent. Without
this, RLHF-trained models often generate plausible-sounding answers even when the context is empty.

**Citation format forces structured output.** Demanding `[page N]` citation strings in a
predictable bracket format allows downstream code (`re.finditer(r"\[page (\d+)"`) to parse them
reliably. Unstructured "see page 3" or "(p.3)" references are harder to parse and validate.

**Context before question prevents "lost in the middle" degradation.** Transformer attention
weights degrade for tokens in the middle of a long context window. Placing the question at the
end ensures it's in the recency-favoured region. Context chunks are the longest part; they go
first so any degradation hits them rather than the question or the grounding rule.

---

## Phase 05 — FastAPI Backend

Phase 05 wraps the RAG pipeline in a production FastAPI app. Schemas and routes live in
`apps/api/` (not in `src/mrta/`) because they are HTTP-layer concerns that evolve on a
different cadence to the domain model.

### What's extracted — Phase 05

| Tutorial cell | Inline code | Production file |
|---------------|-------------|-----------------|
| [3] | `class AskRequest`, `class AskResponse`, `class UploadResponse`, `class DocumentInfo` | `apps/api/schemas/ask.py`, `apps/api/schemas/upload.py`, `apps/api/schemas/documents.py` |
| [5] | `api_src = r'''...'''` (full main.py as string) | `apps/api/main.py` (lifespan + router wiring) |
| [5] | `/ask` route inline in `api_src` | `apps/api/routers/ask.py` |
| [5] | `/upload` route inline in `api_src` | `apps/api/routers/upload.py` |
| [5] | `/documents` route inline in `api_src` | `apps/api/routers/documents.py` |

### What's still inline — Phase 05

| Cell | Content | Why it stays |
|------|---------|--------------|
| [8] | `httpx` alive check | Requires a live server; teaching demo |
| [10] | Upload a PDF via httpx | Live server call; teaches the HTTP interface |
| [12] | Ask a question via httpx | Live server call; teaches the HTTP interface |
| [14] | List documents via httpx | Live server call; teaches the HTTP interface |

### Running notebook cell status

| Cell | Status | Notes |
|------|--------|-------|
| [5] | `# Full implementation: see apps/api/schemas/` | Replaced inline definitions |
| [7] | `# Full implementation: see apps/api/main.py` | Replaced inline api_src string |
| [8] | Inline — keep | httpx alive check; skips gracefully if server not running |
| [10] | Inline — keep | Upload demo |
| [12] | Inline — keep | Ask demo |
| [14] | Inline — keep | Documents demo |

### Routes implemented — Phase 05

| Method | Path | Response model | Handler |
|--------|------|----------------|---------|
| `GET` | `/health` | `dict[str, str]` | `apps/api/main.py` |
| `POST` | `/ask` | `AskResponse` | `apps/api/routers/ask.py` |
| `POST` | `/upload` | `UploadResponse` | `apps/api/routers/upload.py` |
| `GET` | `/documents` | `list[DocumentInfo]` | `apps/api/routers/documents.py` |

### Dependency injection pattern

Shared resources (vector store, LLM client) are loaded once at startup via the lifespan
context manager and stored on `app.state`. Routes receive them through FastAPI's `Depends`
mechanism via two functions in `apps/api/deps.py`:

```python
def get_store(request: Request): return request.app.state.store
def get_llm(request: Request): return request.app.state.llm
```

This is the swap point for tests: `app.dependency_overrides[get_store] = lambda: mock_store`
replaces the real store with a mock in `tests/unit/test_api.py` without touching production code.

### Interface refinements — tutorial → production

**`VectorStore(dim=..., embedder=...)` → `VectorStore(embedder)`**

The tutorial used the old two-arg constructor. Production uses `Embedder()` (which reads
`settings.embedding_model`) and passes the embedder object directly — `dim` is derived
from the embedder internally.

**`build_pipeline(store)` / `RagPipeline.run(...)` → `rag_query(...)`**

The tutorial imported a `build_pipeline` function and `RagPipeline` class that do not exist
in the production library. The `/ask` route calls `rag_query(question, vector_store=store, llm=llm, top_k=k)`
directly — it is the single entry point defined in `src/mrta/core/rag_pipeline.py`.

### Concept note — why API schemas live in apps/api/ not src/mrta/core/schemas.py

Two separate versioning boundaries exist in this project:

- **Library schemas** (`src/mrta/core/schemas.py`): `PageRecord`, `PdfDocument`, `Chunk` — evolve
  with the domain model. A change here means re-chunking, re-embedding, or changing retrieval logic.

- **API schemas** (`apps/api/schemas/`): `AskRequest`, `AskResponse`, `UploadResponse`, `DocumentInfo`
  — evolve with the HTTP contract. A change here means updating client code (the Streamlit app,
  any external consumers).

Keeping them separate means a library refactor (e.g. adding a field to `Chunk`) does not force a
breaking API change, and vice versa. The `/ask` route translates between the two layers: it receives
an `AskRequest`, calls `rag_query` which returns `list[Chunk]`, and maps each `Chunk` to a
`SourceChunk` for the response.
