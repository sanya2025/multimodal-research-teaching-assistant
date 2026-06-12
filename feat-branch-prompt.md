# Feature Branch Prompt

Paste this at the start of any `feat/` branch session, replacing `<feature-name>` and the
placeholders below.

---

## feat/`<feature-name>` — session start

I am working on branch `feat/<feature-name>`.

Read these files first before proposing or writing anything:

1. `production-ready.md` — bottom section `## feature/<feature-name>` (if it exists) gives the
   agreed Understanding, design, and step-by-step plan.
2. `CHANGELOG.md` — top entry shows the expected changed-files table and test table for this
   feature.
3. Every source file listed under **Relevant files** in the `production-ready.md` entry.
4. Every test file listed in the CHANGELOG entry.

---

## Which markdown files to update — and when

### Before implementation (plan phase)

| File | What to add | Where |
|------|-------------|-------|
| `production-ready.md` | New `## feature/<feature-name>` section | Append at the bottom |
| `CHANGELOG.md` | New `## [feat/<feature-name>]` entry | Prepend at the top (above all existing entries) |

#### `production-ready.md` section template

```markdown
## feature/<feature-name>

### Understanding

**Current implementation:**

- <what exists today — stubs, bare raises, missing wrapping, etc.>

**Relevant files:**

- `<path>` — <what to do>

**Risks:**

- <wrapping too broadly, breaking changes, silent fallbacks to leave alone, etc.>

### Design

<hierarchy diagram, schema, interface, or key decisions — depends on feature type>

### Steps

**1 — <first atomic step>**

<code snippet if needed>

**2 — <second step>**

...

### Expected outcome

- <what passes after implementation>
- <what is importable / callable>
- <test count>
```

#### `CHANGELOG.md` entry template

```markdown
## [feat/<feature-name>] — <Short Title> — <YYYY-MM-DD>

**Commit:** `TBD`

<One-paragraph summary: what changed and why.>

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `<path>` | Created / Updated | <what changed> |

### Tests created — `tests/unit/test_<feature>.py` (<N> tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `Test<X>` | `test_<name>` | <what it asserts> |
```

---

### After implementation (close-out phase)

| File | What to update | Detail |
|------|---------------|--------|
| `CHANGELOG.md` | Replace `TBD` commit hash | Run `git log --oneline -1` after merge to get hash |
| `production-ready.md` library map | Change `stub` → `✅ done` | For every module that was a stub and is now complete |
| `notebook-to-production-steps.md` | Add a row to the session log | Only if this feature touched a notebook |
| `docs/adr/` | Create a new ADR | Only if the feature introduced a significant architectural decision (new dep, new pattern, swap of component) |

---

## Rules

- Never write to `production-ready.md` or `CHANGELOG.md` without reading them first.
- Append `production-ready.md` entries at the **bottom**; prepend `CHANGELOG.md` entries at
  the **top**.
- Keep `production-ready.md` entries focused on design intent — not session narrative.
- Keep `CHANGELOG.md` entries factual — changed files and tests only, no opinions.
- Do not update `README.md` unless the public API or install instructions changed.
- Do not create an ADR unless a non-obvious architectural decision was made.

---

---

## chore/docker-healthchecks — session start

I am working on branch `chore/docker-healthchecks`.

Read these files first before proposing or writing anything:

1. `production-ready.md` — bottom section `## chore/docker-healthchecks` gives the agreed
   Understanding, design, and step-by-step plan.
2. `CHANGELOG.md` — top entry shows the expected changed-files table for this chore.
3. `docker/docker-compose.yml` — current service definitions and bare `depends_on` blocks.
4. `docker/Dockerfile.api` — current API image definition.
5. `docker/Dockerfile.streamlit` — current Streamlit image definition.

---

## Which markdown files to update — and when

### Before implementation (plan phase)

| File | What to add | Where |
|------|-------------|-------|
| `production-ready.md` | New `## chore/docker-healthchecks` section | Append at the bottom |
| `CHANGELOG.md` | New `## [chore/docker-healthchecks]` entry | Prepend at the top |

#### `production-ready.md` section to add

````markdown
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
| ollama (compose) | 15s | 5s | 60s | 5 |
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
`depends_on` to condition-based:

```yaml
  ollama:
    image: ollama/ollama:latest
    ...
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 15s
      timeout: 5s
      start_period: 60s
      retries: 5

  api:
    ...
    depends_on:
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 15s
      timeout: 5s
      start_period: 30s
      retries: 3

  streamlit:
    ...
    depends_on:
      api:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]
      interval: 15s
      timeout: 5s
      start_period: 30s
      retries: 3
```

### Expected outcome

- `docker compose up` starts services in dependency order and waits for each to pass its
  healthcheck before starting the next.
- `docker compose ps` shows `(healthy)` for all three services once fully up.
- API never attempts an Ollama call before Ollama is ready.
- No new tests — this is pure Docker configuration.

```

#### `CHANGELOG.md` entry to add

```markdown
## [chore/docker-healthchecks] — Docker Healthchecks & Startup Ordering — 2026-06-11

**Commit:** `TBD`

Adds `HEALTHCHECK` instructions to both Dockerfiles and `healthcheck` blocks to
`docker-compose.yml`. Upgrades bare `depends_on` to condition-based (`service_healthy`)
so the API waits for Ollama to be ready before starting, and Streamlit waits for the
API. Fixes unreliable `docker compose up` in local development.

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `docker/Dockerfile.api` | Updated | `HEALTHCHECK` added — polls `GET /health` via Python urllib |
| `docker/Dockerfile.streamlit` | Updated | `HEALTHCHECK` added — polls `GET /_stcore/health` via Python urllib |
| `docker/docker-compose.yml` | Updated | `healthcheck` blocks on all 3 services; `depends_on` upgraded to `condition: service_healthy` |

### No new test files

Pure Docker/Compose configuration — no library or API code changed.
```

---

### After implementation (close-out phase)

| File | What to update | Detail |
|------|---------------|--------|
| `CHANGELOG.md` | Replace `TBD` with merge commit hash | `git log --oneline -1` after merge |
| `production-ready.md` | Replace "Expected outcome" with "Actual outcome (shipped)" | Note any timing tweaks made during testing |

---

## Rules

- Read `docker-compose.yml` and both Dockerfiles before writing anything.
- Do not change any `src/mrta/` or `apps/` code — only Docker files.
- Do not create an ADR — this is configuration, not an architectural decision.
- Test locally with `docker compose up --build` and confirm `docker compose ps` shows `(healthy)` before marking done.

---

---

## feat/cross-encoder-reranking — session start

I am working on branch `feat/cross-encoder-reranking`.

Read these files first before proposing or writing anything:

1. `production-ready.md` — bottom section `## feat/cross-encoder-reranking` (if it exists)
   gives the agreed Understanding, design, and step-by-step plan.
2. `CHANGELOG.md` — top entry shows the expected changed-files table and test table for this
   feature.
3. `src/mrta/retrieval/vector_store.py` — the `search()` method is the insertion point.
4. `src/mrta/core/rag_pipeline.py` — `rag_query()` is where the reranker is wired in.
5. `src/mrta/core/schemas.py` — `Chunk` is the data type flowing through retrieval and reranking.
6. `tests/unit/test_rag_pipeline.py` — existing pipeline tests that must continue to pass.

---

## Which markdown files to update — and when

### Before implementation (plan phase)

| File | What to add | Where |
|------|-------------|-------|
| `production-ready.md` | New `## feat/cross-encoder-reranking` section | Append at the bottom |
| `CHANGELOG.md` | New `## [feat/cross-encoder-reranking]` entry | Prepend at the top |

#### `production-ready.md` section to add

```markdown
## feat/cross-encoder-reranking

### Understanding

**Current implementation:**

- `src/mrta/retrieval/` has `embedder.py` and `vector_store.py` — no `reranker.py`.
- `src/mrta/retrieval/__init__.py` exports only `Embedder` and `VectorStore`.
- `src/mrta/core/rag_pipeline.py` — `rag_query()` calls `vector_store.search(question, k=top_k)`
  and passes the results directly to the prompt. No reranking step exists.
- Cross-encoder reranking is a two-stage pattern: vector search retrieves a broad candidate
  set (top-k), then a cross-encoder model scores each (query, chunk) pair and re-orders them
  by relevance. Only the top-n highest-scored chunks are passed to the LLM.

**Relevant files:**

- `src/mrta/retrieval/reranker.py` — create: `Reranker` class wrapping `CrossEncoder`
- `src/mrta/retrieval/__init__.py` — export `Reranker`
- `src/mrta/core/rag_pipeline.py` — add optional `reranker` and `rerank_top_n` parameters
- `pyproject.toml` — add `sentence-transformers` to a new `[reranker]` optional extra
- `tests/unit/test_reranker.py` — create: unit tests for `Reranker` (mock CrossEncoder)

**Dependencies:**

- `sentence-transformers` provides `CrossEncoder` — add as optional extra `[reranker]` in
  `pyproject.toml` to keep the default install lightweight.
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (small, fast, MRC-tuned).
- `CrossEncoder` must be mocked in tests — the real model download must not run in CI.

**Risks:**

- `rag_query()` signature change adds parameters — existing callers (API, tests) must still
  work when `reranker=None` (the default). Do not break the no-reranker path.
- Model download at construction time makes the `Reranker` expensive to instantiate in tests.
  Patch `sentence_transformers.CrossEncoder.__init__` and `predict` in the test.
- `rerank_top_n` must be ≤ `top_k` — if the caller passes a larger value, return as many
  chunks as available rather than raising. Keep it silent, not a hard error.

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

### Steps

**1 — `src/mrta/retrieval/reranker.py`** — implement `Reranker`:

```python
from __future__ import annotations
from mrta.core.schemas import Chunk

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 3) -> list[Chunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_n]]
```

**2 — `src/mrta/retrieval/__init__.py`** — add `Reranker` to exports:

```python
from mrta.retrieval.reranker import Reranker
__all__ = ["Embedder", "VectorStore", "Reranker"]
```

**3 — `src/mrta/core/rag_pipeline.py`** — add optional reranker parameters:

```python
def rag_query(
    question: str,
    vector_store: VectorStore,
    llm: LLMClient,
    top_k: int = 5,
    reranker: Reranker | None = None,
    rerank_top_n: int = 3,
) -> dict:
    ...
    sources = vector_store.search(question, k=top_k)
    if reranker is not None:
        sources = reranker.rerank(question, sources, top_n=rerank_top_n)
    ...
```

**4 — `pyproject.toml`** — add optional extra:

```toml
[project.optional-dependencies]
reranker = ["sentence-transformers>=3.0"]
```

**5 — `tests/unit/test_reranker.py`** — write unit tests (see Expected outcome).

### Expected outcome

- `from mrta.retrieval import Reranker` works.
- `rag_query(..., reranker=None)` behaves identically to the current implementation.
- `rag_query(..., reranker=reranker, rerank_top_n=3)` calls `reranker.rerank()` and passes
  the reranked chunks to the prompt instead of the raw vector-search results.
- All existing `test_rag_pipeline.py` tests pass unchanged.
- `tests/unit/test_reranker.py` — 8–10 tests covering: basic rerank, score ordering,
  empty input, `top_n` > len(chunks) safety, `rag_query` integration with mock reranker.
```

#### `CHANGELOG.md` entry to add

```markdown
## [feat/cross-encoder-reranking] — Cross-Encoder Reranking for RAG — 2026-06-11

**Commit:** `TBD`

Adds optional cross-encoder reranking to the RAG pipeline. `rag_query()` now accepts
a `reranker` parameter; when provided, vector-search candidates are re-scored by a
`CrossEncoder` model and only the top-n highest-relevance chunks are passed to the LLM.
Improves answer precision without affecting callers that omit the parameter.

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/retrieval/reranker.py` | Created | `Reranker` class wrapping `sentence-transformers` `CrossEncoder` |
| `src/mrta/retrieval/__init__.py` | Updated | Added `Reranker` to `__all__` |
| `src/mrta/core/rag_pipeline.py` | Updated | `rag_query()` gains `reranker` and `rerank_top_n` optional params |
| `pyproject.toml` | Updated | New `[reranker]` optional extra: `sentence-transformers>=3.0` |
| `tests/unit/test_reranker.py` | Created | Unit tests for `Reranker` with mocked `CrossEncoder` |

### Tests created — `tests/unit/test_reranker.py` (~10 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestReranker` | `test_rerank_returns_top_n` | Returns exactly `top_n` chunks |
| `TestReranker` | `test_rerank_orders_by_score_descending` | Higher-scored chunk appears first |
| `TestReranker` | `test_rerank_empty_input` | Returns empty list without error |
| `TestReranker` | `test_rerank_top_n_exceeds_chunks` | Returns all chunks when `top_n` > len |
| `TestReranker` | `test_rerank_calls_predict_with_pairs` | `CrossEncoder.predict` receives `(query, text)` pairs |
| `TestRagPipelineReranking` | `test_rag_query_calls_reranker_when_provided` | `reranker.rerank` is called once |
| `TestRagPipelineReranking` | `test_rag_query_skips_reranker_when_none` | `reranker` is not called |
| `TestRagPipelineReranking` | `test_rag_query_passes_reranked_chunks_to_prompt` | Prompt receives reranked chunks, not raw |
```

---

### After implementation (close-out phase)

| File | What to update | Detail |
|------|---------------|--------|
| `CHANGELOG.md` | Replace `TBD` with merge commit hash | `git log --oneline -1` after merge |
| `production-ready.md` library map | Change `retrieval/reranker` stub → `✅ done` | If it was listed as stub |
| `docs/adr/` | Create `ADR-007-cross-encoder-reranking.md` | New dependency (`sentence-transformers`) and new pipeline stage warrant an ADR |

---

## Rules

- Read `rag_pipeline.py`, `vector_store.py`, and `schemas.py` before writing anything.
- Do not change the `Chunk` schema — the reranker works with existing `Chunk` objects.
- Do not make reranking mandatory — `reranker=None` must keep the pipeline working exactly as before.
- Mock `CrossEncoder` in all tests — never download a model in CI.
- Create ADR-007 after implementation — new dep and new pipeline stage qualify.

---

---

## test/rag-evaluation-gates — session start

I am working on branch `test/rag-evaluation-gates`.

Read these files first before proposing or writing anything:

1. `production-ready.md` — bottom section `## test/rag-evaluation-gates` (if it exists)
   gives the agreed Understanding, design, and step-by-step plan.
2. `CHANGELOG.md` — top entry shows the expected changed-files table and test table for this
   branch.
3. `src/mrta/evaluation/metrics.py` — existing 4 metrics; this file is extended (not replaced).
4. `src/mrta/evaluation/eval_pipeline.py` — `run_eval()` calls a live LLM + VectorStore; read
   it to understand the boundary between library metrics and end-to-end eval. Do NOT change it.
5. `src/mrta/core/schemas.py` — `Chunk` and `EvalReport` are the shared data types.
6. `tests/evaluation/datasets/golden_qa.yaml` — 30 questions; fields are `id`, `question`,
   `expected_answer`, `expected_documents`, `expected_keywords`. The `source` field is
   documented in the YAML header but absent from individual questions — loader must use
   `.get("source", "")` safely.
7. `tests/unit/test_metrics.py` — existing tests for the 4 current metrics; must still pass.

---

## Which markdown files to update — and when

### Before implementation (plan phase)

| File | What to add | Where |
|------|-------------|-------|
| `production-ready.md` | New `## test/rag-evaluation-gates` section | Append at the bottom |
| `CHANGELOG.md` | New `## [test/rag-evaluation-gates]` entry | Prepend at the top |

#### `production-ready.md` section to add

````markdown
## test/rag-evaluation-gates

### Understanding

**Current implementation:**

- `src/mrta/evaluation/metrics.py` has `answer_relevance`, `faithfulness`,
  `citation_correctness`, `hallucination_rate` — no Recall@k, MRR, nDCG@10, or
  retrieval-side source coverage.
  **Important distinction:** `citation_correctness` (existing) is answer-side — it checks
  whether `[page N]` citation patterns in the generated answer refer to real pages in the
  retrieved chunks. The new `citation_coverage` function is retrieval-side — it checks
  whether the right source documents were retrieved at all. They measure different things;
  do not confuse or merge them.
- `src/mrta/evaluation/eval_pipeline.py` calls real `rag_query()` with a live LLM and
  VectorStore — cannot run in CI without deployed services. Do not change it.
- `tests/evaluation/` exists but has no `__init__.py`, no tests, and no baselines.
- `tests/evaluation/datasets/golden_qa.yaml` — 30 questions with `expected_documents`
  (PDF filename list) and `expected_keywords` (keyword list). No `source` field on items.
- `pyproject.toml` — `testpaths = ["tests"]` already covers `tests/evaluation/**` recursively.
  No CI or pyproject change needed.

**Relevant files:**

- `src/mrta/evaluation/metrics.py` — add `recall_at_k`, `mrr`, `ndcg_at_k`, `citation_coverage`
- `tests/evaluation/__init__.py` — create (package marker)
- `tests/evaluation/conftest.py` — create: `load_golden_qa()` + `golden_qa` pytest fixture
- `tests/evaluation/test_retrieval_metrics.py` — create: unit tests for the 3 new functions
- `tests/evaluation/test_rag_gates.py` — create: threshold gate tests using golden QA + mocked
  retrieval
- `tests/evaluation/test_citation_coverage.py` — create: citation coverage tests using golden
  QA `expected_keywords`
- `tests/evaluation/baselines/retrieval_metrics.json` — create: starting thresholds
- `tests/evaluation/README.md` — create: documentation

**Dependencies:**

- `PyYAML>=6.0` already in core deps — no new deps.
- `pytest>=8.2.0` already in `[dev]` — no new deps.
- No LLM calls, no network access, no model downloads — all tests must be deterministic.

**Risks:**

- `eval_pipeline.py` imports `LLMClient` and calls real `rag_query()`. Do NOT import it
  in the new evaluation tests — only test the metric functions directly.
- Adding 3 new functions to `metrics.py` must not break `tests/unit/test_metrics.py`.
- The `source` field is absent from golden QA items — use `.get("source", "")` in loader.

### Design

**New metric functions** — add to `src/mrta/evaluation/metrics.py`:

```python
def recall_at_k(retrieved_sources: list[str], expected_docs: list[str], k: int) -> float:
    """Fraction of expected_docs found in the first k entries of retrieved_sources."""
    top = retrieved_sources[:k]
    if not expected_docs:
        return 1.0
    return sum(1 for d in expected_docs if d in top) / len(expected_docs)


def mrr(retrieved_sources: list[str], expected_docs: list[str]) -> float:
    """Reciprocal rank of the first retrieved_source that is in expected_docs."""
    expected_set = set(expected_docs)
    for rank, src in enumerate(retrieved_sources, start=1):
        if src in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_sources: list[str], expected_docs: list[str], k: int = 10) -> float:
    """nDCG@k — rewards placing expected_docs near the top of retrieved_sources."""
    import math
    expected_set = set(expected_docs)
    top = retrieved_sources[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, src in enumerate(top, start=1)
        if src in expected_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected_docs), k) + 1))
    return dcg / ideal if ideal > 0 else 1.0


def citation_coverage(retrieved_sources: list[str], expected_docs: list[str]) -> float:
    """Retrieval-side: fraction of expected_docs found anywhere in retrieved_sources (no k cutoff)."""
    if not expected_docs:
        return 1.0
    retrieved_set = set(retrieved_sources)
    return sum(1 for d in expected_docs if d in retrieved_set) / len(expected_docs)
```

Note: All four functions take `list[str]` where `retrieved_sources = [c.source for c in
retrieved_chunks]`. `Chunk.source` stores the PDF filename (e.g., `"CLIP.pdf"`), matching
`expected_documents` in the golden QA.

**Two citation metrics — keep them separate:**

| Function | Side | What it measures |
|----------|------|-----------------|
| `citation_coverage` (new) | Retrieval | Did the retriever surface the right source documents? |
| `citation_correctness` (existing) | Answer | Does the generated answer cite real page numbers? |

**Golden QA loader** — `tests/evaluation/conftest.py`:

```python
from pathlib import Path
import pytest
import yaml

GOLDEN_QA_PATH = Path(__file__).parent / "datasets" / "golden_qa.yaml"


def load_golden_qa(path: Path = GOLDEN_QA_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    return [
        {
            "id": q["id"],
            "question": q["question"],
            "expected_answer": q.get("expected_answer", ""),
            "expected_documents": q.get("expected_documents", []),
            "expected_keywords": q.get("expected_keywords", []),
            "source": q.get("source", ""),
        }
        for q in data.get("questions", [])
    ]


@pytest.fixture(scope="session")
def golden_qa() -> list[dict]:
    return load_golden_qa()
```

**Baseline thresholds** — `tests/evaluation/baselines/retrieval_metrics.json`:

```json
{
  "recall_at_5":       { "baseline": 0.85, "tolerance": 0.05 },
  "recall_at_10":      { "baseline": 0.90, "tolerance": 0.05 },
  "mrr":               { "baseline": 0.70, "tolerance": 0.05 },
  "ndcg_at_10":        { "baseline": 0.75, "tolerance": 0.05 },
  "citation_coverage": { "baseline": 0.90, "tolerance": 0.05 }
}
```

Gate assertion pattern: `assert score >= entry["baseline"] - entry["tolerance"]`.
A 5% regression from baseline fails CI; small natural variation within tolerance passes.
To tighten a gate after measuring the real retriever, update `"baseline"` — do not reduce
`"tolerance"` without deliberate decision.

### Steps

**1 — `src/mrta/evaluation/metrics.py`** — append `recall_at_k`, `mrr`, `ndcg_at_k`,
`citation_coverage`; add all four to `__all__`. `ndcg_at_k` uses `import math` inside the
function body — no top-level import needed.

**2 — `tests/evaluation/__init__.py`** — create as empty file.

**3 — `tests/evaluation/conftest.py`** — implement `load_golden_qa()` and `golden_qa` fixture
as shown above.

**4 — `tests/evaluation/test_retrieval_metrics.py`** — ~12 pure unit tests for the 4 new
metric functions; no golden QA, no YAML, no Chunk objects. Test edge cases: empty inputs,
k=0, no overlap, partial overlap, exact match at boundary k. For `ndcg_at_k` specifically:
verify that a relevant doc at rank 1 produces a higher score than the same doc at rank 3.

**5 — `tests/evaluation/test_citation_coverage.py`** — ~5 tests using golden QA items:
load the dataset, construct fake `Chunk` sources matching `expected_documents`, verify
`citation_coverage` score is correct. Add a TODO comment explaining how to replace fake
chunks with real VectorStore output.

**6 — `tests/evaluation/test_rag_gates.py`** — ~6 tests: load
`tests/evaluation/baselines/retrieval_metrics.json`, construct a "perfect" mock retrieval
(all `expected_documents` at the top of the result list), compute each metric, and assert
`score >= entry["baseline"] - entry["tolerance"]`. Add a TODO comment per test explaining
how to replace mock retrieval with `VectorStore.search()` output after indexing. Include one
`test_baselines_file_structure` test that verifies every key in the JSON has both `"baseline"`
and `"tolerance"` fields. These tests verify the gate mechanism is wired correctly, not that
the real retriever currently meets the thresholds.

**7 — `tests/evaluation/baselines/retrieval_metrics.json`** — create with starting thresholds.

**8 — `tests/evaluation/README.md`** — document: purpose of golden QA set, how to run
`pytest tests/evaluation`, how to update thresholds, how to add new questions, why PDFs are
not committed.

### Expected outcome

- `pytest tests/evaluation` passes with no network calls and no model downloads.
- All 4 new metric functions importable: `from mrta.evaluation.metrics import recall_at_k, mrr, ndcg_at_k, citation_coverage`.
- `tests/unit/test_metrics.py` continues to pass unchanged.
- ~22 new deterministic tests across 3 test files.
- `tests/evaluation/baselines/retrieval_metrics.json` committed with delta-based thresholds
  (`baseline` + `tolerance` per metric).
- `tests/evaluation/README.md` present and accurate.
````

#### `CHANGELOG.md` entry to add

````markdown
## [test/rag-evaluation-gates] — RAG Evaluation Gates — 2026-06-11

**Commit:** `TBD`

Adds deterministic retrieval evaluation gates. Extends `metrics.py` with four retrieval
metrics: `recall_at_k`, `mrr`, `ndcg_at_k`, and `citation_coverage` (retrieval-side source
coverage; distinct from the existing answer-side `citation_correctness`). Wires the golden
QA dataset into a pytest fixture and three test files. Gate tests load delta-based thresholds
from `baselines/retrieval_metrics.json` — each entry has `baseline` and `tolerance` — and
assert `score >= baseline − tolerance`, allowing small natural variation while failing on
regressions. No LLM calls, no network access.

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/evaluation/metrics.py` | Updated | Added `recall_at_k`, `mrr`, `ndcg_at_k`, `citation_coverage` (retrieval-side) |
| `tests/evaluation/__init__.py` | Created | Package marker |
| `tests/evaluation/conftest.py` | Created | `load_golden_qa()` + `golden_qa` session fixture |
| `tests/evaluation/test_retrieval_metrics.py` | Created | Unit tests for 3 new metric functions |
| `tests/evaluation/test_citation_coverage.py` | Created | Citation coverage tests with golden QA |
| `tests/evaluation/test_rag_gates.py` | Created | Threshold gate tests with mocked retrieval |
| `tests/evaluation/baselines/retrieval_metrics.json` | Created | Starting thresholds |
| `tests/evaluation/README.md` | Created | Evaluation documentation |

### Tests created — `tests/evaluation/` (~22 tests)

| Test file | Test | Assertion |
|-----------|------|-----------|
| `test_retrieval_metrics.py` | `test_recall_at_k_perfect` | All expected docs in top-k → 1.0 |
| `test_retrieval_metrics.py` | `test_recall_at_k_partial` | Half in top-k → 0.5 |
| `test_retrieval_metrics.py` | `test_recall_at_k_none` | No match → 0.0 |
| `test_retrieval_metrics.py` | `test_recall_at_k_respects_cutoff` | Doc at rank k+1 not counted |
| `test_retrieval_metrics.py` | `test_recall_at_k_empty_expected` | Empty expected → 1.0 |
| `test_retrieval_metrics.py` | `test_mrr_first_rank` | Relevant at rank 1 → 1.0 |
| `test_retrieval_metrics.py` | `test_mrr_second_rank` | Relevant at rank 2 → 0.5 |
| `test_retrieval_metrics.py` | `test_mrr_no_match` | No relevant doc → 0.0 |
| `test_retrieval_metrics.py` | `test_citation_coverage_full` | All expected docs in retrieved → 1.0 |
| `test_retrieval_metrics.py` | `test_citation_coverage_partial` | Some missing → fractional |
| `test_retrieval_metrics.py` | `test_ndcg_at_k_rank1_beats_rank3` | Relevant doc at rank 1 > rank 3 |
| `test_retrieval_metrics.py` | `test_ndcg_at_k_empty_expected` | Empty expected → 1.0 |
| `test_citation_coverage.py` | `test_coverage_clip_question` | CLIP.pdf in sources → 1.0 |
| `test_citation_coverage.py` | `test_coverage_wrong_source` | BLIP-2.pdf for CLIP question → 0.0 |
| `test_citation_coverage.py` | `test_coverage_empty_expected` | Empty expected → 1.0 |
| `test_citation_coverage.py` | `test_load_golden_qa_returns_all_items` | Loader returns 30 items |
| `test_citation_coverage.py` | `test_loader_handles_missing_optional_fields` | Missing source → "" |
| `test_rag_gates.py` | `test_recall_at_5_gate` | score ≥ baseline − tolerance (0.85 − 0.05) |
| `test_rag_gates.py` | `test_recall_at_10_gate` | score ≥ baseline − tolerance (0.90 − 0.05) |
| `test_rag_gates.py` | `test_mrr_gate` | score ≥ baseline − tolerance (0.70 − 0.05) |
| `test_rag_gates.py` | `test_ndcg_at_10_gate` | score ≥ baseline − tolerance (0.75 − 0.05) |
| `test_rag_gates.py` | `test_citation_coverage_gate` | score ≥ baseline − tolerance (0.90 − 0.05) |
| `test_rag_gates.py` | `test_baselines_file_structure` | Every key has `"baseline"` and `"tolerance"` |
````

---

### After implementation (close-out phase)

| File | What to update | Detail |
|------|---------------|--------|
| `CHANGELOG.md` | Replace `TBD` with merge commit hash | `git log --oneline -1` after merge |
| `production-ready.md` | Mark `evaluation/metrics.py` functions as `✅ done` | If listed as stubs |

---

## Rules

- Read `metrics.py`, `eval_pipeline.py`, `schemas.py`, and `golden_qa.yaml` before writing anything.
- Do not change `eval_pipeline.py` — it calls a live LLM and is not part of this branch.
- Do not import `eval_pipeline` or `LLMClient` in any `tests/evaluation/` file.
- All tests must be deterministic — no network calls, no model downloads, no Ollama.
- New functions in `metrics.py` use `list[str]` (sources), not `list[Chunk]` — keep them
  pure and easy to unit-test.
- Do not create an ADR — no new dependencies and no architectural decision.
- Do not change `README.md` — no public API or install instructions changed.

---

## feat/opentelemetry-tracing — session start

I am working on branch `feat/opentelemetry-tracing`.

Read these files first before proposing or writing anything:

1. `production-ready.md` — bottom section `## feat/opentelemetry-tracing` for the agreed plan.
2. `CHANGELOG.md` — top entry for the expected changed-files and test tables.
3. `src/mrta/observability/logging.py` — existing observability pattern; match its style.
4. `src/mrta/observability/__init__.py` — current exports; extend, don't replace.
5. `src/mrta/core/config.py` — `Settings` class; `enable_tracing: bool = False` already exists;
   three new OTEL fields need adding.
6. `src/mrta/core/rag_pipeline.py` — primary instrumentation target; read before touching.
7. `src/mrta/ingestion/chunker.py` — secondary instrumentation target (`chunk_pdf`).
8. `src/mrta/evaluation/eval_pipeline.py` — tertiary target (`run_eval`); calls live LLM —
   do not change its signature or behavior, only wrap with spans.

---

## Which markdown files to update — and when

### Before implementation (plan phase)

| File | What to add | Where |
|------|-------------|-------|
| `production-ready.md` | New `## feat/opentelemetry-tracing` section | Append at the bottom |
| `CHANGELOG.md` | New `## [feat/opentelemetry-tracing]` entry | Prepend at the top |

---

### Understanding

**Current implementation:**

- `src/mrta/observability/` has `logging.py` (`StructuredLogger`) and `__init__.py` only.
  `tracing.py` does not exist.
- `src/mrta/core/config.py` `Settings` has `enable_tracing: bool = False` but no OTEL
  endpoint, service name, or console exporter fields.
- `src/mrta/core/rag_pipeline.py` `rag_query()` times its own execution but has no spans.
- `src/mrta/ingestion/chunker.py` `chunk_pdf()` has no observability.
- `src/mrta/evaluation/eval_pipeline.py` `run_eval()` has no observability.
- `pyproject.toml` has no `opentelemetry-*` dependencies in any group.

**Relevant files:**

- `src/mrta/observability/tracing.py` — create: tracer init, `get_tracer()`, `trace_span()`
- `src/mrta/observability/__init__.py` — extend: add `get_tracer`, `trace_span` to `__all__`
- `src/mrta/core/config.py` — extend: add `otel_service_name`, `otel_exporter_otlp_endpoint`,
  `otel_console_exporter` to `Settings`
- `src/mrta/core/rag_pipeline.py` — extend: add spans inside `rag_query()`
- `src/mrta/ingestion/chunker.py` — extend: add `mrta.ingestion` span inside `chunk_pdf()`
- `src/mrta/evaluation/eval_pipeline.py` — extend: add `mrta.evaluation` span inside `run_eval()`
- `pyproject.toml` — extend: add `[otel]` optional dep group; add OTEL API + SDK to `[dev]`
- `tests/unit/test_tracing.py` — create: ~10 tests using `InMemorySpanExporter`
- `docs/observability.md` — create: how to enable and interpret traces
- `docs/adr/ADR-007-opentelemetry-tracing.md` — create: why OTEL, why optional, why SDK not SaaS

**Dependencies:**

- `opentelemetry-api>=1.24` — tracer API; no-op by default when SDK not configured.
  Required in base deps (needed at import time even without tracing enabled).
- `opentelemetry-sdk>=1.24` — configures exporters; only activated when `enable_tracing=True`.
  Add to `[dev]` (needed for tests) and `[otel]` (needed for production use).
- `opentelemetry-exporter-otlp-proto-grpc>=1.24` — OTLP/gRPC export to Jaeger/Tempo.
  Add to `[otel]` only — not required for local console tracing.

**Risks:**

- `opentelemetry-api` must go in base deps (not optional) because `rag_pipeline.py` and
  `chunker.py` will call `get_tracer()` at the call site. If it were optional, importing
  those modules without the dep installed would raise `ImportError`. `opentelemetry-api`
  is a pure-Python, zero-dependency package — safe as a base dep.
- Do not call `TracerProvider` configuration inside `rag_query()` or any hot path. Call
  `configure_tracer()` once at process startup (from the API lifespan or `__main__`).
- `eval_pipeline.py` calls a live LLM. Only wrap `run_eval()` with a span — do not change
  its signature, behavior, or imports.
- `configure_tracer()` must be idempotent — calling it twice should not register duplicate
  exporters. Use a module-level `_configured: bool` guard.

### Design

**`src/mrta/observability/tracing.py`** — full interface:

```python
"""mrta.observability.tracing — optional OpenTelemetry instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.trace import Span

_configured: bool = False


def configure_tracer(
    service_name: str = "mrta",
    console: bool = False,
    otlp_endpoint: str = "",
) -> None:
    """Set up the OTEL SDK once. Call at process startup when enable_tracing=True.

    console=True → logs spans to stdout (useful for local dev).
    otlp_endpoint → exports spans via OTLP/gRPC to Jaeger, Tempo, etc.
    """
    global _configured
    if _configured:
        return
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    if console:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str = "mrta") -> trace.Tracer:
    """Return the module-level tracer. No-op proxy when not configured."""
    return trace.get_tracer(name)


@contextmanager
def trace_span(
    span_name: str,
    attributes: dict | None = None,
) -> Generator[Span, None, None]:
    """Context manager that wraps a block in a named span.

    Attributes are set after span creation. Exceptions are recorded and re-raised.
    No-op when tracing is not configured.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
```

Note: all SDK imports are lazy (inside `configure_tracer`) so the module is importable
without `opentelemetry-sdk` installed — only `opentelemetry-api` is required.

**Three new `Settings` fields** (add in the `# --- Observability ---` block of `config.py`):

```python
otel_service_name: str = "mrta"
otel_exporter_otlp_endpoint: str = ""   # empty = no OTLP export
otel_console_exporter: bool = False
```

**Span names and attributes per operation:**

| Span name | Where | Key attributes |
|-----------|-------|----------------|
| `mrta.rag_query` | `rag_pipeline.rag_query()` | `query.length`, `retrieval.top_k`, `reranker.enabled`, `reranker.top_n`, `retrieval.chunk_count`, `model.llm`, `latency_ms` |
| `mrta.ingestion` | `chunker.chunk_pdf()` | `document.path`, `chunk.strategy`, `chunk.count` |
| `mrta.evaluation` | `eval_pipeline.run_eval()` | `benchmark.size` |

**Startup wiring** — in `apps/api/main.py` lifespan (read-only guidance, not a file to change
in this branch unless it is trivial — see Rules):

```python
if settings.enable_tracing:
    from mrta.observability.tracing import configure_tracer
    configure_tracer(
        service_name=settings.otel_service_name,
        console=settings.otel_console_exporter,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
    )
```

**ADR-007** — create `docs/adr/ADR-007-opentelemetry-tracing.md`. Cover:

- Why OTEL (vendor-neutral, industry standard, no SaaS lock-in)
- Why optional (avoids forced dependency on a backend for local dev and CI)
- Why API as base dep + SDK as optional dep (keeps import graph clean)
- Alternatives considered: structlog-only, custom timing dict, Jaeger client direct

### Steps

**1 — `pyproject.toml`** — add `opentelemetry-api>=1.24` to base `dependencies`; add
`opentelemetry-sdk>=1.24` to `[dev]`; add a new `[otel]` group with
`opentelemetry-api>=1.24`, `opentelemetry-sdk>=1.24`,
`opentelemetry-exporter-otlp-proto-grpc>=1.24`.

**2 — `src/mrta/observability/tracing.py`** — implement exactly as shown in Design above.
Module-level `_configured` guard. All SDK imports are lazy inside `configure_tracer()`.

**3 — `src/mrta/observability/__init__.py`** — add `get_tracer` and `trace_span` to imports
and `__all__`.

**4 — `src/mrta/core/config.py`** — add the three new `otel_*` fields to `Settings` in the
`# --- Observability ---` block, after `enable_tracing`.

**5 — `src/mrta/core/rag_pipeline.py`** — wrap `rag_query()` body in a `trace_span`
context manager for `mrta.rag_query`; set attributes after `vector_store.search()` returns
(so `retrieval.chunk_count` is known) and after the full function completes. Import
`trace_span` lazily inside the function body to avoid circular imports.

**6 — `src/mrta/ingestion/chunker.py`** — wrap `chunk_pdf()` body in `trace_span` for
`mrta.ingestion`; set `document.path`, `chunk.strategy`, `chunk.count`. Lazy import.

**7 — `src/mrta/evaluation/eval_pipeline.py`** — wrap `run_eval()` body in `trace_span`
for `mrta.evaluation`; set `benchmark.size = len(benchmark)`. Lazy import. Do not change
function signature or return type.

**8 — `tests/unit/test_tracing.py`** — ~10 tests using `InMemorySpanExporter` from
`opentelemetry.sdk.trace.export.in_memory_span_exporter`. Reset `_configured` between
tests using `monkeypatch` on `mrta.observability.tracing._configured`. Verify:

- `get_tracer()` returns a tracer without crashing when not configured
- `trace_span` yields a span and sets attributes
- `trace_span` records an exception and re-raises it
- `configure_tracer(console=True)` registers a `ConsoleSpanExporter`
- `configure_tracer` is idempotent (calling twice does not register two exporters)
- spans appear in `InMemorySpanExporter` after a traced block

**9 — `docs/observability.md`** — create: what is traced, how to enable console tracing
locally (`OTEL_CONSOLE_EXPORTER=true` + `ENABLE_TRACING=true`), how to point at Jaeger,
what attributes to look for in spans, why tracing helps debug RAG latency.

**10 — `docs/adr/ADR-007-opentelemetry-tracing.md`** — create ADR as described in Design.

### Expected outcome

- `from mrta.observability.tracing import get_tracer, trace_span, configure_tracer` works.
- All existing tests continue to pass — tracing is no-op when not configured.
- `ENABLE_TRACING=true OTEL_CONSOLE_EXPORTER=true` prints spans to stdout for `rag_query`.
- ~10 new tests in `tests/unit/test_tracing.py`, all deterministic (no network, no Ollama).
- `docs/observability.md` explains how to use tracing locally.
- `docs/adr/ADR-007-opentelemetry-tracing.md` exists and is linked from `docs/adr/`.

---

#### `CHANGELOG.md` entry to add

````markdown
## [feat/opentelemetry-tracing] — OpenTelemetry Tracing — 2026-06-12

**Commit:** `TBD`

Adds optional OpenTelemetry tracing to the RAG lifecycle. Creates
`src/mrta/observability/tracing.py` with `configure_tracer()`, `get_tracer()`, and
`trace_span()`. Instruments `rag_query()`, `chunk_pdf()`, and `run_eval()` with named spans
and structured attributes. Tracing is disabled by default (`enable_tracing=False`) and
requires no external backend — enable console output with `OTEL_CONSOLE_EXPORTER=true`.
Three new `otel_*` settings fields added. `opentelemetry-api` added to base deps (no-op
proxy when SDK not configured); `opentelemetry-sdk` added to `[dev]` and new `[otel]` group.

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/observability/tracing.py` | Created | `configure_tracer`, `get_tracer`, `trace_span` |
| `src/mrta/observability/__init__.py` | Updated | Added `get_tracer`, `trace_span` to `__all__` |
| `src/mrta/core/config.py` | Updated | Added `otel_service_name`, `otel_exporter_otlp_endpoint`, `otel_console_exporter` |
| `src/mrta/core/rag_pipeline.py` | Updated | `mrta.rag_query` span with query/retrieval/model attributes |
| `src/mrta/ingestion/chunker.py` | Updated | `mrta.ingestion` span with path/strategy/count attributes |
| `src/mrta/evaluation/eval_pipeline.py` | Updated | `mrta.evaluation` span with benchmark size |
| `pyproject.toml` | Updated | `opentelemetry-api` in base deps; `opentelemetry-sdk` in `[dev]`; new `[otel]` group |
| `tests/unit/test_tracing.py` | Created | ~10 tests using `InMemorySpanExporter` |
| `docs/observability.md` | Created | How to enable and interpret traces |
| `docs/adr/ADR-007-opentelemetry-tracing.md` | Created | ADR for OTEL adoption |

### Tests created — `tests/unit/test_tracing.py` (~10 tests)

| Test | Assertion |
|------|-----------|
| `test_get_tracer_returns_tracer` | `get_tracer()` returns a `Tracer` without error |
| `test_trace_span_yields_span` | `trace_span` context manager yields without error |
| `test_trace_span_sets_attributes` | Attributes appear in exported span |
| `test_trace_span_records_exception` | Exception is recorded on span and re-raised |
| `test_configure_tracer_console` | `ConsoleSpanExporter` registered after configure |
| `test_configure_tracer_idempotent` | Second `configure_tracer` call does not add extra processors |
| `test_spans_captured_by_in_memory_exporter` | Span name appears in `InMemorySpanExporter` |
| `test_trace_span_noop_when_not_configured` | No crash when SDK not configured |
| `test_existing_rag_tests_unaffected` | `rag_query` with mock LLM still passes (import smoke) |
````

---

### After implementation (close-out phase)

| File | What to update | Detail |
|------|---------------|--------|
| `CHANGELOG.md` | Replace `TBD` with merge commit hash | `git log --oneline -1` after merge |
| `production-ready.md` | Mark `observability/tracing.py` as `✅ done` | It is currently a stub |
| `docs/adr/` README or index | Link `ADR-007` | If an ADR index file exists |

---

## Rules

- Read all 8 files listed under **Read these files first** before writing anything.
- `opentelemetry-api` must be in base deps (not optional) — `rag_pipeline.py` and
  `chunker.py` will call `get_tracer()` unconditionally.
- All SDK imports (`TracerProvider`, `ConsoleSpanExporter`, etc.) must be lazy — inside
  `configure_tracer()` only. This keeps `tracing.py` importable without `opentelemetry-sdk`.
- Do not change `eval_pipeline.py` function signature — only wrap the body with a span.
- Do not add tracing to `apps/api/main.py` lifespan in this branch unless it is a
  one-liner addition that does not risk breaking the API tests.
- All tests must be deterministic — no network calls, no Ollama, no real OTLP endpoint.
- `_configured` must be reset between tests using `monkeypatch` — never rely on test
  execution order.
- Create ADR-007 — OpenTelemetry is a non-obvious architectural dependency decision.
- Do not change `README.md`.
