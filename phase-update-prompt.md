# Phase Update Prompt

Paste this at the start of any notebook-to-production session, replacing `NN` and `<module>`.

## Phase 05 prompt

Convert Phase 05 (FastAPI Backend) tutorial notebook to production.

Read these four files first:

1. `production-ready.md` → Phase 05 section — confirms: schemas in `apps/api/schemas/`,
   routes in `apps/api/routers/`, thin-adapter rule (no business logic in routes)
2. `notebook-to-production-steps.md` → Phase 04 is the last completed phase
3. Tutorial notebook: `notebooks/tutorials/2026-05-25-phase05-fastapi-backend.ipynb`
4. Production notebook: `notebooks/production/2026-05-25-phase05-fastapi-backend.ipynb`

Also inspect `apps/api/` before writing anything — the scaffold already exists:

- `apps/api/main.py` — minimal FastAPI app with `/health` only; lifespan and routes not yet wired
- `apps/api/schemas/__init__.py` — empty stub
- `apps/api/routers/__init__.py` — empty stub
- No individual router or schema files yet

Two interface refinements — tutorial inline code does not match the current production APIs:

- Tutorial lifespan uses raw `SentenceTransformer(...)` and `VectorStore(dim=..., embedder=...)`
  (old interface) → Production: `Embedder()` + `VectorStore(embedder)` (no separate `dim` param)
- Tutorial uses `build_pipeline(store=store)` and `RagPipeline.run(...)` (these do not exist) →
  Production: call `rag_query(question, vector_store=store, llm=llm, top_k=k)` directly in the
  `/ask` route handler

5. No new library schemas — `AskRequest`, `AskResponse`, `UploadResponse`, `DocumentInfo` belong
   in `apps/api/schemas/`, not in `src/mrta/core/schemas.py`. `Chunk` from `mrta.core.schemas`
   is already available and used inside route logic.

6a. Create `apps/api/schemas/ask.py`:

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)

class SourceChunk(BaseModel):
    page: int
    chunk_id: str
    preview: str  # first 200 chars of Chunk.text

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_s: float
```

Drop `cited_pages`, `retrieved`, `model` from the tutorial's `AskResponse` — they don't align
with `rag_query`'s return. Use `top_k` (not `k`) to match `rag_query`'s parameter name.

6b. Create `apps/api/schemas/upload.py`:

```python
class UploadResponse(BaseModel):
    doc_id: str
    source: str
    n_pages: int
    n_chunks: int
```

6c. Create `apps/api/schemas/documents.py`:

```python
class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    n_pages: int
    n_chunks: int
```

6d. Update `apps/api/schemas/__init__.py` — re-export all schemas:
`AskRequest`, `AskResponse`, `SourceChunk`, `UploadResponse`, `DocumentInfo`.

6e. Create `apps/api/routers/ask.py` — `POST /ask`:

```python
@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, store=Depends(get_store), llm=Depends(get_llm)) -> AskResponse:
    result = rag_query(req.question, vector_store=store, llm=llm, top_k=req.top_k)
    sources = [
        SourceChunk(page=c.page, chunk_id=c.chunk_id, preview=c.text[:200])
        for c in result["sources"]
    ]
    return AskResponse(answer=result["answer"], sources=sources, latency_s=result["latency_s"])
```

6f. Create `apps/api/routers/upload.py` — `POST /upload`:
Validate `.pdf` extension (raise `HTTPException(400, ...)` otherwise); save to `data/raw/`;
call `load_pdf(path)`; call `chunk_pdf(pdf, strategy="recursive")`; call `store.add(chunks)`;
persist with `store.save(settings.vector_store_path / "default")`;
return `UploadResponse(doc_id=..., source=..., n_pages=..., n_chunks=len(chunks))`.

6g. Create `apps/api/routers/documents.py` — `GET /documents`:
Aggregate `store._chunks` (a `list[Chunk]`) by `doc_id` — use `.doc_id`, `.source`, `.page`
attributes (not dict keys; the tutorial's `state["store"].metadata` was the old dict API).
Return `list[DocumentInfo]`.

6h. Update `apps/api/main.py`:

- Add lifespan context manager: create `Embedder()`; load `VectorStore` from
  `settings.vector_store_path / "default"` if it exists, otherwise create a fresh one;
  create `LLMClient()`; store all three on `app.state`
- Add dependency functions `get_store(request: Request)` and `get_llm(request: Request)` that
  read from `request.app.state`
- Include all three routers; keep the existing `/health` endpoint

7. Update `apps/api/routers/__init__.py` — import the three router modules so they are
   discoverable (or leave empty — main.py includes them directly).

8. Production notebook — cells [5] and [7] print the `apps/api/main.py` source as a string.
   Replace each with a single-line comment: `# Full implementation: see apps/api/main.py`
   Cells [10], [12], [14], [16] (the `httpx` demo calls) keep inline — they require a live
   server and are the teaching content of this notebook. Update cell [1] header from
   "Production note" to "Production note (active)".

9. Write tests in `tests/unit/test_api.py`:
   - Use `fastapi.testclient.TestClient` — no live server, no live Ollama needed
   - Override dependencies with `app.dependency_overrides` to inject a mock store and mock llm
   - `GET /health` returns 200 and `{"status": "ok"}`
   - `POST /ask` with a valid payload returns 200 and response has `answer` and `sources` keys
   - `POST /ask` with a question shorter than 3 chars returns 422 (Pydantic validation)
   - `GET /documents` with a populated mock store returns a list of `DocumentInfo` dicts
   - `POST /upload` with `tests/fixtures/sample.pdf` returns 200 with `doc_id`, `n_pages`,
     `n_chunks` — mock `chunk_pdf` to avoid the embedding model during this test

10. Run `MRTA_ENV=test pytest -q` — all tests must pass.

Also extend CI lint targets in `.github/workflows/ci.yml` so `apps/` is linted alongside the
library:

```yaml
- name: Lint (ruff)
  run: ruff check src/ tests/ apps/

- name: Format check (black)
  run: black --check src/ tests/ apps/
```

Update documents:

11. `production-ready.md` — Phase 05 table: mark `AskRequest`/`AskResponse`, `UploadResponse`,
    `DocumentInfo`, `/ask`, `/upload`, `/documents` routes → `✅ done`.

12. `notebook-to-production-steps.md` → add Phase 05 section with: "What's extracted" table
    (tutorial cell → production file in `apps/api/`), "What's still inline" table (httpx demo
    cells [10]–[16]), running notebook cell status table, "Routes implemented" table
    (method + path + response model), and a concept note on why API schemas live in
    `apps/api/schemas/` and not `src/mrta/core/schemas.py` (versioning boundary: library schemas
    evolve with the domain model; API schemas evolve with the HTTP contract — decoupling them
    means a library refactor doesn't force a breaking API change and vice versa).

Wrap up: update memory, suggest git commit commands.

After committing, update `CHANGELOG.md`: read the file, run `git log --oneline -10` for the
Phase 05 hash, read the Phase 05 section of `notebook-to-production-steps.md` and
`tests/unit/test_api.py`. Insert a new entry at the top (above `## [Phase 04]`):
`## [Phase 05] — FastAPI Backend — YYYY-MM-DD` with commit hash, notebook paths,
changed files table (columns `File | Change | Notes`), and tests table
(columns `Test class | Test | Assertion`). Do not modify existing entries.
Suggest: `git add CHANGELOG.md && git commit -m "docs: update CHANGELOG for Phase 05"`

---

## Phase 04 prompt

Convert Phase 04 (End-to-End RAG Pipeline) tutorial notebook to production.

Read these four files first:

1. `production-ready.md` → Phase 04 section — confirms: extract `LLMClient`, `rag_query`,
   `load_prompt` + `rag.j2`, and `StructuredLogger`; interfaces defined there
2. `notebook-to-production-steps.md` → Phase 03 is the last completed phase
3. Tutorial notebook: `notebooks/tutorials/2026-05-25-phase04-rag-pipeline.ipynb`
4. Production notebook: `notebooks/production/2026-05-25-phase04-rag-pipeline.ipynb`
   — cell [1] lists the target imports; cells [4], [6], [8], [10], [16] are still inline

Two interface refinements — tutorial inline code does not match production-ready.md, these are
deliberate upgrades:

- Tutorial `OllamaLLM.chat(system, user)` →
  Production `LLMClient.chat(messages: list[dict], temperature: float = 0.1) -> str`
  where `messages` is `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
  and the return value is plain `str` (not a dict).
- Tutorial `rag_answer` returns `{"retrieved": [{"page": ..., "score": ..., "chunk_id": ...}]}` →
  Production `rag_query` returns `{"answer": str, "sources": list[Chunk], "latency_s": float}`.

Cell [4] in the production notebook also needs updating — Phase 03 is now done. It still
re-defines `VectorStore` inline and creates a raw `SentenceTransformer`. Replace with:

```python
from mrta.retrieval import Embedder, VectorStore
embedder = Embedder()
store = VectorStore.load("data/vector_store/aiayn", embedder)
```

5. No new schemas — `Chunk` is already in `src/mrta/core/schemas.py`

6a. Create `src/mrta/core/llm.py` — `LLMClient`:

```python
class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        # reads settings.llm_provider and settings.ollama_llm_model by default
    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        # Ollama-only for now; returns response text
```

Implement only the Ollama provider. Return plain `str` (response content only, not a dict).

6b. Create `src/mrta/core/rag_pipeline.py` — `rag_query`:

```python
def rag_query(
    question: str,
    vector_store: VectorStore,
    llm: LLMClient,
    top_k: int = 5,
) -> dict:
    """Returns {"answer": str, "sources": list[Chunk], "latency_s": float}"""
```

- Call `vector_store.search(question, k=top_k)` → `sources: list[Chunk]`
- Build prompt: `load_prompt("rag", chunks=sources, question=question)`
- Call `llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])`
- Parse `[page N]` citations from the answer text with `re.finditer`

6c. Create `src/mrta/prompts/__init__.py` — `load_prompt`:

```python
from jinja2 import Environment, PackageLoader
_env = Environment(loader=PackageLoader("mrta", "prompts"))

def load_prompt(name: str, **kwargs: object) -> str:
    """Render src/mrta/prompts/{name}.j2 with kwargs."""
    return _env.get_template(f"{name}.j2").render(**kwargs)
```

`PackageLoader("mrta", "prompts")` uses files at `src/mrta/prompts/` directly with an editable
install. If `pyproject.toml` has no `package-data` entry for `*.j2`, add one before testing.

6d. Create `src/mrta/prompts/rag.j2` — copy the template body from tutorial cell [4]
(the string inside `Template(r"""...""")`). It already uses `{{ c.source }}`, `{{ c.page }}`,
`{{ c.text }}` — these match `Chunk` attribute names, so no edits to the template are needed.

6e. Create `src/mrta/observability/logging.py` — `StructuredLogger`:

```python
class StructuredLogger:
    def log_run(
        self,
        question: str,
        answer: str,
        sources: list[Chunk],
        latency_s: float,
    ) -> None:
        """Append one JSON line to settings.log_file."""
```

Reads `settings.log_file`; creates parent directories; appends with `open(..., "a")`.

7. Update exports:
   - `src/mrta/observability/__init__.py` — export `StructuredLogger`
   - `src/mrta/__init__.py` — add `LLMClient`, `rag_query`, `load_prompt`, `StructuredLogger`
     to imports and `__all__`

8. Update production notebook — replace inline blocks with imports:
   - Cell [4]: replace inline `VectorStore` + `SentenceTransformer` with
     `from mrta.retrieval import Embedder, VectorStore`; update load call to use `Embedder()`
   - Cell [6]: replace `from jinja2 import Template` + `RAG_PROMPT = Template(...)` with
     `from mrta.prompts import load_prompt`; update demo print to `load_prompt("rag", ...)`
   - Cell [8]: replace `class OllamaLLM:` with `from mrta.core.llm import LLMClient`;
     `llm = LLMClient()  # reads settings.llm_provider and settings.ollama_llm_model`
   - Cell [10]: replace `def rag_answer(...)` with `from mrta.core.rag_pipeline import rag_query`
   - Cell [12]: keep inline — update to `rag_query(question, vector_store=store, llm=llm)`;
     access `out["sources"]` (list[Chunk], use `.page`) instead of `out["retrieved"]`
   - Cell [14]: keep inline — update loop to use `rag_query` and `out["sources"]`
   - Cell [16]: replace `def log_run(...)` with `from mrta.observability.logging import StructuredLogger`;
     `logger = StructuredLogger()`; update demo call to `logger.log_run(...)`
   - Cell [1]: update header from "Production imports" → "Production imports (active)"

9. Write tests in `tests/unit/test_rag_pipeline.py`:
   - Use `unittest.mock.patch("mrta.core.llm.ollama.chat")` to mock Ollama — no live LLM needed
   - `LLMClient.chat(messages)` returns the mocked text
   - `rag_query(...)` returns a dict with `answer`, `sources`, `latency_s` keys
   - `sources` are `Chunk` instances
   - `rag_query` with `top_k=1` returns exactly 1 source
   - `load_prompt("rag", chunks=[], question="test")` returns non-empty string containing "test"
   - `StructuredLogger.log_run(...)` appends exactly one line to the log file (use `tmp_path`)
   - The appended JSON line contains `question` and `answer` keys

10. Run `MRTA_ENV=test pytest -q` — all tests must pass

Update documents:

11. `production-ready.md` — library map: `llm.py`, `rag_pipeline.py`, `prompts/`,
    `observability/logging.py` → `✅ done`; Phase 04 table: mark all rows ✅ done

12. `notebook-to-production-steps.md` → add Phase 04 section with: "What's extracted" table,
    "What's still inline" table (cells [12], [14], [17]), running notebook cell status table,
    "Classes and functions implemented" table with method signatures, and a concept note on why
    the RAG prompt separates system role + context block + question into three distinct sections
    (grounding rule prevents hallucination; citation format forces structured output; separation
    prevents "lost in the middle" degradation).

Wrap up: update memory, suggest git commit commands.

After committing, update `CHANGELOG.md`: read the file, run `git log --oneline -10` for the
Phase 04 hash, read the Phase 04 section of `notebook-to-production-steps.md` and
`tests/unit/test_rag_pipeline.py`. Insert a new entry at the top (above `## [Phase 03]`):
`## [Phase 04] — End-to-End RAG Pipeline — YYYY-MM-DD` with commit hash, notebook paths,
changed files table (columns `File | Change | Notes`), and tests table
(columns `Test class | Test | Assertion`). Do not modify existing entries.
Suggest: `git add CHANGELOG.md && git commit -m "docs: update CHANGELOG for Phase 04"`

---

## Phase 03 prompt

Convert Phase 03 (Embeddings & FAISS) tutorial notebook to production.

## Before starting

1. Read `production-ready.md` → Phase 03 section — confirms: extract Embedder class and
   VectorStore class; interfaces defined there
2. Read `notebook-to-production-steps.md` → Phase 02 is the last completed phase
3. Tutorial notebook: `notebooks/tutorials/2026-05-25-phase03-embeddings-and-faiss.ipynb`
4. Production notebook: `notebooks/production/2026-05-25-phase03-embeddings-and-faiss.ipynb`
   — cell [1] already lists the target imports; cells [4], [6], [14] are still inline

## Implement

### Important: two different extraction patterns in this phase

- `VectorStore` EXISTS inline in tutorial cell [14] — extract and refine it
- `Embedder` does NOT exist inline — the tutorial uses `SentenceTransformer` directly;
  `Embedder` is a NEW wrapper class designed to the interface in `production-ready.md`
- `VectorStore.search()` currently returns `list[dict]`; refine it to return `list[Chunk]`

5. No new schemas needed — `Chunk` is already in `src/mrta/core/schemas.py`

6a. Create `src/mrta/retrieval/embedder.py`:
    ```python
    class Embedder:
        def __init__(self, model_name: str | None = None) -> None:
            # reads settings.embedding_model if model_name is None
        def embed(self, texts: list[str]) -> np.ndarray: ...  # float32, L2-normalised
        @property
        def dim(self) -> int: ...
    ```
    Model selection reads `settings.embedding_model` by default; `model_name` overrides it.
    This is why test.yaml uses `all-MiniLM-L6-v2` and dev.yaml uses `nomic-embed-text`.

6b. Create `src/mrta/retrieval/vector_store.py` — extract from tutorial cell [14]:
    - `__init__(self, embedder: Embedder)` — drop explicit `dim` param; use `embedder.dim`
    - `add(self, chunks: list[Chunk]) -> None`
    - `search(self, query: str, k: int = 5) -> list[Chunk]`  ← returns Chunk, not dict
    - `save(self, path: Path) -> None` — writes index.faiss + metadata.jsonl + config.json
    - `load(cls, path: Path, embedder: Embedder) -> VectorStore`

7. Create `src/mrta/retrieval/__init__.py`:

    ```python
    from mrta.retrieval.embedder import Embedder
    from mrta.retrieval.vector_store import VectorStore
    __all__ = ["Embedder", "VectorStore"]
    ```

   Export `Embedder` and `VectorStore` from `src/mrta/__init__.py` and add to `__all__`

8. Update production notebook — replace inline blocks with imports:
   - Cell [4]: replace inline Chunk definition with
     `from mrta.core.schemas import Chunk`
     (keep the JSONL loading lines — they demonstrate how chunks are loaded)
   - Cell [6]: replace inline `SentenceTransformer(...)` instantiation with
     `from mrta.retrieval.embedder import Embedder`
     `embedder = Embedder()   # reads settings.embedding_model`
     `DIM = embedder.dim`
   - Cell [8]: keep inline — teaches the raw embedding API
   - Cell [10]: keep inline — teaches the raw FAISS query API
   - Cell [14]: replace inline VectorStore class definition with
     `from mrta.retrieval.vector_store import VectorStore`
     (keep the store.add / store.save / reloaded.search demo lines)
   - Cells [16], [18]: keep inline — HNSW demo and sanity checks are demonstrations

9. Write tests in `tests/unit/test_vector_store.py`:
   - Load fixture PDF → chunk with `fixed_chunks` → build `VectorStore`
   - `search` returns exactly k results
   - Each result is a `Chunk` instance
   - `save` / `load` round-trip: reloaded store returns same top result for same query
   - `Embedder`: `embed(["hello"])` returns shape `(1, dim)`, norms ~1.0
   - Use `pytest.importorskip("faiss")` to skip gracefully if faiss not installed

10. Run `MRTA_ENV=test pytest -q` — all tests must pass

## Update documents

11. `production-ready.md`:
    - Library map: update `embedder.py` and `vector_store.py` lines from `stub` → `✅ done`
    - Phase 03 table: mark all rows ✅ done

12. `notebook-to-production-steps.md` → add Phase 03 section:
    - "What's extracted" table (tutorial cell → production import)
    - "What's still inline" table (cells [8], [10], [16], [18] — teaching demos)
    - Running notebook cell status table
    - "Classes implemented" table (Embedder + VectorStore with method signatures)
    - Concept note: why normalize embeddings + use IndexFlatIP inner product
      (cosine similarity on unit vectors = dot product; no FAISS training needed at this scale)

## Wrap up

13. Update memory (`save to memory`)
14. Suggest git commit commands

---

## Phase 02 prompt

Convert Phase 02 (Chunking Strategies) tutorial notebook to production.

### Before starting

1. Read `production-ready.md` → Phase 02 section — confirms: extract Chunk schema,
   fixed_chunks, recursive_chunks, token_chunks, semantic_chunks, chunk_pdf dispatcher
2. Read `notebook-to-production-steps.md` → Phase 01 is the last completed phase
3. Tutorial notebook: `notebooks/tutorials/2026-05-25-phase02-chunking-strategies.ipynb`
4. Production notebook: `notebooks/production/2026-05-25-phase02-chunking-strategies.ipynb`
   — cell [1] already lists the target imports; cells [4–17] are all still inline

### Implement

5. Add `Chunk` schema to `src/mrta/core/schemas.py`:

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

## Phase prompt template

```markdown
Convert Phase NN (<name>) tutorial notebook to production.

## Before starting
1. Read `production-ready.md` → Phase NN section — confirm what to extract, target files, interfaces
2. Read `notebook-to-production-steps.md` → check where Phase NN-1 left off
3. Read the tutorial notebook (`notebooks/tutorials/...`) — understand the inline code
4. Read the production notebook (`notebooks/production/...`) — see what stubs are already there

## Implement
5. Add any new schemas to `src/mrta/core/schemas.py`
6. Implement `src/mrta/<module>.py` — extract functions from tutorial notebook, add type hints + docstrings
7. Export new public symbols in `src/mrta/__init__.py` (`__all__`)
8. Activate the production notebook imports — replace inline definitions with `from mrta.X import Y`
9. Write tests in `tests/unit/test_<module>.py`
10. Run `MRTA_ENV=test pytest -q` — all tests must pass

## Update documents
11. `production-ready.md` → mark extracted items ✅ done in Phase NN table; update library map line
12. `notebook-to-production-steps.md` → add Phase NN section:
    - "What's extracted" table (tutorial cell → production import)
    - "What's still inline" table (if anything remains)
    - Running notebook cell status table
    - "Functions implemented" table (signature + what it does)
    - Concept note for any non-obvious technique introduced

## Wrap up
13. Update memory (`save to memory`)
14. Suggest git commit commands
```

---

## CHANGELOG update prompt

Update `CHANGELOG.md` to reflect the most recently completed phase.

Read these files before writing anything:

1. `CHANGELOG.md` — note the most recent phase entry and its exact table format.
2. Run `git log --oneline -10` — find commit(s) added since the last CHANGELOG entry; record
   the short hash and message.
3. `notebook-to-production-steps.md` → locate the section for the phase just completed —
   use it as the source of truth for what was extracted and what stayed inline.
4. `tests/unit/test_{module}.py` for the phase — list every test function and its assertion.

Insert a new entry at the top of `CHANGELOG.md`, immediately below the title paragraph and
above the most recent `## [Phase NN]` heading. Use this structure:

- Heading: `## [Phase NN] — {Name} — YYYY-MM-DD`
- Metadata lines: commit hash, tutorial notebook path, production notebook path.
- Changed files table — columns `File | Change | Notes`. One row per file touched (source
  module, `__init__.py` exports, production notebook, test file, both doc files). "Change"
  is `Created` or `Updated` — be exact. "Notes" is one sentence on what changed and why.
- Tests table — columns `Test class | Test | Assertion`. One row per test function. "Assertion"
  describes what is actually verified. If no new test file was added, one sentence explaining why.

Do not modify any existing entry. Newest phase stays at the top. Dates as YYYY-MM-DD.

Suggest commit: `git add CHANGELOG.md && git commit -m "docs: update CHANGELOG for Phase NN"`
