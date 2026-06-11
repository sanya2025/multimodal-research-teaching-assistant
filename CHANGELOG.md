# Changelog

Records notebook-to-production conversions and significant library changes.
Each entry maps tutorial notebook cells → `src/mrta/` modules → production notebook → unit tests.

---

## [chore/docker-healthchecks] — Docker Healthchecks & Startup Ordering — 2026-06-11

**Commit:** `a62b56b`

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

---

## [chore/ci-quality-gates] — CI Quality Gates — 2026-06-10

**Commit:** `5d3a5d7`

Strengthens CI from a single lint+test job into four parallel jobs: the existing `test`
job plus `type-check` (mypy), `audit` (pip-audit), and `docker` (build + smoke test).
Each new job runs only after `test` passes, keeping the fast-fail guarantee.

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `.github/workflows/ci.yml` | Updated | 3 new jobs: `type-check` (mypy), `audit` (pip-audit), `docker` (build + smoke test); pip upgraded before audit to clear PYSEC-2026-196; retry loop on `/health` with container logs on failure |
| `pyproject.toml` | Updated | `mypy>=1.10.0` and `pip-audit>=2.7.0` added to `dev` group; `[tool.mypy]` section added |
| `docker/Dockerfile.api` | Updated | Added `COPY LICENSE ./` and `COPY README.md ./` — hatchling requires both at build time |
| `src/mrta/core/config.py` | Updated | Fixed `settings_customise_sources` signature to match pydantic-settings supertype (mypy) |
| `src/mrta/retrieval/embedder.py` | Updated | `TYPE_CHECKING` guard for `SentenceTransformer` import; `_st_model` typed as `SentenceTransformer \| None`; `_ensure_st` return type added (mypy) |
| `src/mrta/retrieval/vector_store.py` | Updated | `_index` typed as `faiss.Index` instead of `faiss.IndexFlatIP` to match `faiss.read_index` return type (mypy) |
| `apps/api/routers/upload.py` | Updated | `filename = file.filename or ""` guard added — `file.filename` is `str \| None` (mypy) |

### No new test files

CI/tooling configuration plus pre-existing mypy annotation gaps surfaced by the new `type-check` job. All 119 existing tests continue to pass.

---

## [feat/upload-validation] — Upload Validation & Hardening — 2026-06-10

**Commit:** `ffa5171`

Hardens the `POST /upload` endpoint with five ordered validation layers: extension check,
20 MB size limit, PDF magic-byte check, safe filename sanitisation, and structured error
handling for malformed PDFs. Corrupt or oversized files now return explicit 4xx responses
instead of propagating as 500 Internal Server Errors.

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `apps/api/routers/upload.py` | Updated | Five-layer validation; reads file once into memory; safe filename via `Path.name` |
| `apps/api/schemas/upload.py` | Updated | `UploadError(BaseModel)` added with `detail` and `code` fields |
| `apps/api/main.py` | Updated | `IngestionError` exception handler → 422 with `code: "malformed_pdf"` |
| `tests/unit/test_api.py` | Updated | 4 new tests added to `TestUpload` — see table below |

### Tests added — `tests/unit/test_api.py` (4 new tests in `TestUpload`)

| Test | Assertion |
|------|-----------|
| `test_oversized_file_returns_413` | File > 20 MB → 413 before any disk write |
| `test_non_pdf_magic_bytes_returns_415` | `.pdf` extension but wrong magic bytes → 415 |
| `test_path_traversal_filename_is_sanitised` | `../../evil.pdf` saved as `evil.pdf` in `data/raw/` |
| `test_malformed_pdf_returns_422` | `IngestionError` from `load_pdf` → 422, not 500 |

---

## [feat/exception-hierarchy] — Exception Hierarchy — 2026-06-10

**Commit:** `d68464f`

Library-wide exception hierarchy replacing bare `ValueError` / `RuntimeError` raises and
unwrapped third-party errors at every system boundary. All exceptions are catchable as
`MRTAError` or narrowed to a specific subclass.

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/core/exceptions.py` | Created | `MRTAError(Exception)` base + 5 subclasses: `IngestionError`, `EmbeddingError`, `RetrievalError`, `LLMError`, `EvaluationError` |
| `src/mrta/__init__.py` | Updated | All 6 exception classes added to public API and `__all__` |
| `src/mrta/ingestion/chunker.py` | Updated | `ValueError` → `IngestionError` at `chunk_pdf` dispatch |
| `src/mrta/ingestion/pdf_loader.py` | Updated | `fitz.open` wrapped — corrupt/missing PDF raises `IngestionError` |
| `src/mrta/retrieval/embedder.py` | Updated | `httpx.HTTPStatusError` from `raise_for_status` wrapped in `EmbeddingError` |
| `src/mrta/retrieval/vector_store.py` | Updated | `faiss.read_index` failure wrapped in `RetrievalError` |
| `src/mrta/core/llm.py` | Updated | `ollama.chat` failure wrapped in `LLMError` |
| `src/mrta/multimodal/vlm_client.py` | Updated | `ollama.chat` failure wrapped in `LLMError` |
| `tests/unit/test_chunker.py` | Updated | `test_unknown_strategy_raises` updated from `ValueError` → `IngestionError` |
| `tests/unit/test_exceptions.py` | Created | 8 tests — see table below |

### Tests created — `tests/unit/test_exceptions.py` (8 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestHierarchy` | `test_base_is_exception` | `MRTAError` is a subclass of `Exception` |
| `TestHierarchy` | `test_all_subclasses_are_mrta_error` | All 5 subclasses are subclasses of `MRTAError` |
| `TestRaiseSites` | `test_pdf_loader_raises_ingestion_error` | `fitz.open` failure → `IngestionError`; `__cause__` set |
| `TestRaiseSites` | `test_embedder_raises_embedding_error` | `httpx.HTTPStatusError` → `EmbeddingError`; `__cause__` set |
| `TestRaiseSites` | `test_vector_store_raises_retrieval_error` | `faiss.read_index` failure → `RetrievalError`; `__cause__` set |
| `TestRaiseSites` | `test_llm_raises_llm_error` | `ollama.chat` failure → `LLMError`; `__cause__` set |
| `TestRaiseSites` | `test_vlm_client_raises_llm_error` | `ollama.chat` failure → `LLMError`; `__cause__` set |
| `TestExport` | `test_importable_from_mrta` | All 6 exception classes importable from top-level `mrta` |

---

## [Phase 09] — Evaluation, Observability & Docker — 2026-06-08

**Commit:** `7c0774d`
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase09-evaluation-logging-docker.ipynb`
**Production notebook:** `notebooks/production/2026-05-25-phase09-evaluation-logging-docker.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/core/schemas.py` | Updated | `EvalReport` model added — `n_questions`, `answer_relevance`, `faithfulness`, `citation_correctness`, `hallucination_rate`, `mean_latency_s` |
| `src/mrta/evaluation/metrics.py` | Created | Four deterministic metrics: `answer_relevance`, `faithfulness`, `citation_correctness`, `hallucination_rate`; pure Python, no model downloads |
| `src/mrta/evaluation/eval_pipeline.py` | Created | `run_eval(benchmark, vs, llm, top_k) -> EvalReport`; iterates benchmark, calls `rag_query`, averages all four metrics |
| `src/mrta/evaluation/__init__.py` | Updated | Exports `run_eval`, `answer_relevance`, `faithfulness`, `citation_correctness`, `hallucination_rate` |
| `src/mrta/__init__.py` | Updated | `EvalReport`, `run_eval`, and all four metric functions added to public API |
| `notebooks/production/…phase09….ipynb` | Updated | Cells [1], [6], [8], [12], [16], [18], [21] replaced with library imports or comments pointing to existing Docker/CI files |
| `tests/unit/test_metrics.py` | Created | 17 tests — see table below |
| `production-ready.md` | Updated | Phase 09 library map rows marked ✅ done; `EvalReport` noted in schema row |
| `notebook-to-production-steps.md` | Updated | Phase 09 section appended |

### Tests created — `tests/unit/test_metrics.py` (17 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestAnswerRelevance` | `test_keyword_present` | score is `1.0` when question keyword appears in answer |
| `TestAnswerRelevance` | `test_no_overlap` | score is in `[0, 1]` when there is no keyword overlap |
| `TestAnswerRelevance` | `test_returns_float` | return type is `float` |
| `TestAnswerRelevance` | `test_range` | score is in `[0, 1]` for arbitrary inputs |
| `TestFaithfulness` | `test_grounded_answer` | score is `1.0` when answer text appears in chunks |
| `TestFaithfulness` | `test_empty_answer` | empty answer scores `1.0` (vacuously grounded) |
| `TestFaithfulness` | `test_returns_float` | return type is `float` |
| `TestFaithfulness` | `test_range` | score is in `[0, 1]` for arbitrary inputs |
| `TestFaithfulness` | `test_ungrounded_sentence` | invented claim scores in `[0, 1]` |
| `TestCitationCorrectness` | `test_correct_citation` | `[page 3]` citing a real page scores `1.0` |
| `TestCitationCorrectness` | `test_wrong_page` | `[page 99]` citing a non-existent page scores `0.0` |
| `TestCitationCorrectness` | `test_no_citations` | answer with no `[page N]` patterns scores `1.0` |
| `TestCitationCorrectness` | `test_mixed_citations` | one valid + one invalid citation scores `0.5` |
| `TestCitationCorrectness` | `test_returns_float` | return type is `float` |
| `TestHallucinationRate` | `test_complement_of_faithfulness` | `faithfulness + hallucination_rate == 1.0` within `1e-9` |
| `TestHallucinationRate` | `test_range` | score is in `[0, 1]` |
| `TestHallucinationRate` | `test_returns_float` | return type is `float` |

---

## [Phase 08] — Teaching Modes & Prompt Engineering — 2026-06-08

**Commit:** `e3ca49a`
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase08-teaching-modes-and-prompts.ipynb`
**Production notebook:** `notebooks/production/2026-05-25-phase08-teaching-modes-and-prompts.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/prompts/_base.j2` | Created | Shared parent template with Jinja2 block inheritance — `role`, `behavior`, `format` blocks; grounding rule and citation instruction shared across all modes |
| `src/mrta/prompts/beginner.j2` | Created | Extends `_base.j2`; plain language, analogies, short paragraphs |
| `src/mrta/prompts/expert.j2` | Created | Extends `_base.j2`; graduate-level ML background assumed; precise terminology and equations |
| `src/mrta/prompts/interview.j2` | Created | Extends `_base.j2`; ML system-design interview style with tradeoffs and complexity analysis |
| `src/mrta/prompts/quiz.j2` | Created | Extends `_base.j2`; generates 5 multiple-choice questions with answer key |
| `src/mrta/prompts/lecture_notes.j2` | Created | Extends `_base.j2`; structured notes with Key Terms, Mechanism, Results, Open questions |
| `src/mrta/prompts/explain.j2` | Created | Standalone template (does not extend `_base.j2`); used by `VLMClient.caption` for figure explanation; optional `level` and `question` variables |
| `src/mrta/prompts/__init__.py` | Updated | `MODES` constant added — maps user-facing mode names to template base names; exported in `__all__` |
| `src/mrta/__init__.py` | Updated | `MODES` added to public API |
| `notebooks/production/…phase08….ipynb` | Updated | Cells [1], [6], [8], [10] replaced with library imports; remaining cells kept inline |
| `tests/unit/test_prompts.py` | Created | 15 tests — see table below |
| `production-ready.md` | Updated | Phase 08 table all rows marked ✅ done; library map updated |
| `notebook-to-production-steps.md` | Updated | Phase 08 section appended |

### Tests created — `tests/unit/test_prompts.py` (15 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestLoadPrompt` | `test_rag_contains_question` | `load_prompt("rag", ...)` returns non-empty string containing the question |
| `TestLoadPrompt` | `test_beginner_contains_question` | `load_prompt("beginner", ...)` returns non-empty string containing the question |
| `TestLoadPrompt` | `test_expert_returns_nonempty` | `load_prompt("expert", ...)` returns non-empty string |
| `TestLoadPrompt` | `test_quiz_contains_quiz_marker` | `load_prompt("quiz", ...)` contains `"QUIZ"` |
| `TestLoadPrompt` | `test_interview_returns_nonempty` | `load_prompt("interview", ...)` returns non-empty string |
| `TestLoadPrompt` | `test_lecture_notes_returns_nonempty` | `load_prompt("lecture_notes", ...)` returns non-empty string |
| `TestLoadPrompt` | `test_explain_no_kwargs` | `load_prompt("explain")` returns non-empty string with no required kwargs |
| `TestLoadPrompt` | `test_explain_level_injected` | custom `level` appears in the rendered explain prompt |
| `TestLoadPrompt` | `test_explain_question_injected` | optional `question` kwarg appears in rendered explain prompt |
| `TestLoadPrompt` | `test_unknown_template_raises` | `load_prompt("nonexistent")` raises `jinja2.exceptions.TemplateNotFound` |
| `TestLoadPrompt` | `test_chunks_rendered_in_rag` | RAG prompt includes chunk `source` and `text` content |
| `TestModes` | `test_modes_is_dict` | `MODES` is a `dict` |
| `TestModes` | `test_modes_has_required_keys` | `MODES` contains `beginner`, `expert`, `quiz`, `lecture_notes`, `interview`, `explain` |
| `TestModes` | `test_all_rag_modes_loadable` | every non-explain mode renders to a non-empty string via `load_prompt` |
| `TestModes` | `test_explain_mode_loadable` | `load_prompt(MODES["explain"])` renders to a non-empty string |

---

## [Phase 07] — Figure Extraction & VLM — 2026-06-05

**Commit:** `45e7a3b`
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase07-figure-extraction-and-vlm.ipynb`
**Production notebook:** `notebooks/production/2026-05-25-phase07-figure-extraction-and-vlm.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/core/schemas.py` | Updated | `FigureRecord` model added — `doc_id`, `source`, `page`, `figure_index`, `image_bytes: bytes`; `to_pil()` method converts bytes to `PIL.Image.Image` |
| `src/mrta/ingestion/figure_extractor.py` | Created | `extract_figures(pdf_path) -> list[FigureRecord]`; uses `_doc_id` from `pdf_loader`; stores PNG bytes in-memory — no disk writes |
| `src/mrta/multimodal/clip_embedder.py` | Created | `CLIPEmbedder.embed_image(Image)` / `embed_text(str)`; lazy-imports `open_clip`; returns L2-normalised float32 `(512,)` vectors |
| `src/mrta/multimodal/vlm_client.py` | Created | `VLMClient.caption(Image, prompt)`; reads `settings.ollama_vlm_model`; converts PIL → PNG bytes → base64 → `ollama.chat` |
| `src/mrta/multimodal/__init__.py` | Updated | Exports `CLIPEmbedder`, `VLMClient` |
| `src/mrta/ingestion/__init__.py` | Updated | Adds `extract_figures` alongside `load_pdf`, `chunk_pdf` |
| `src/mrta/__init__.py` | Updated | `FigureRecord`, `extract_figures`, `CLIPEmbedder`, `VLMClient` added to public API |
| `notebooks/production/…phase07….ipynb` | Updated | Cells [1], [5], [9], [13] replaced with library imports; cells [7], [11], [15], [17] kept inline as teaching demos |
| `tests/unit/test_figure_extractor.py` | Created | 5 tests — see table below |
| `tests/unit/test_clip_embedder.py` | Created | 5 tests (skipped if `open_clip` absent) — see table below |
| `production-ready.md` | Updated | Phase 07 table all rows marked ✅ done; library map updated |
| `notebook-to-production-steps.md` | Updated | Phase 07 section appended |

### Tests created — `tests/unit/test_figure_extractor.py` (5 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| — | `test_extract_figures_returns_list` | `extract_figures(sample.pdf)` returns a `list` |
| — | `test_extract_figures_items_are_figure_records` | every item is a `FigureRecord` instance |
| — | `test_figure_record_image_bytes_non_empty` | `image_bytes` is non-empty for every record |
| — | `test_figure_record_page_and_index_positive` | `page >= 1` and `figure_index >= 1` for every record |
| — | `test_to_pil_returns_pil_image` | `to_pil()` returns `PIL.Image.Image` |

### Tests created — `tests/unit/test_clip_embedder.py` (5 tests, skipped if `open_clip` absent)

| Test class | Test | Assertion |
|------------|------|-----------|
| — | `test_embed_image_shape` | `embed_image(1×1 white image)` returns shape `(512,)` |
| — | `test_embed_image_dtype` | embedding dtype is `float32` |
| — | `test_embed_image_l2_norm` | L2 norm is `~1.0` within `1e-5` |
| — | `test_embed_text_shape_and_norm` | `embed_text(str)` returns shape `(512,)` with norm `~1.0` |
| — | `test_image_text_dot_product_positive` | image and matching text embeddings have positive dot product |

---

## [Phase 06] — Streamlit Frontend — 2026-06-05

**Commit:** `087ada3`
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase06-streamlit-frontend.ipynb`
**Production notebook:** `notebooks/production/2026-05-25-phase06-streamlit-frontend.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `apps/streamlit/app.py` | Updated | Full Streamlit UI replacing 7-line scaffold — sidebar upload + doc list, mode radio (6 teaching modes), question input, k slider, answer + cited pages + chunks expander; four API contract fixes: `k` → `top_k`, `cited_pages` derived from `sources`, `model` removed, `retrieved` → `sources` |
| `notebooks/production/…phase06….ipynb` | Updated | Cell [5] replaced with comment pointing to `apps/streamlit/app.py`; cell [1] header → "Production note (active)" |
| `production-ready.md` | Updated | Phase 06 `apps/streamlit/app.py` row marked ✅ done |
| `notebook-to-production-steps.md` | Updated | Phase 06 section appended |

### Tests

No new tests added. Streamlit UI has no unit-testable logic — correctness is covered by the Phase 05 API tests (`tests/unit/test_api.py`).

---

## [Phase 05] — FastAPI Backend — 2026-06-05

**Commit:** `f66426c`
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase05-fastapi-backend.ipynb`
**Production notebook:** `notebooks/production/2026-05-25-phase05-fastapi-backend.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `apps/api/schemas/ask.py` | Created | `AskRequest` (`question`, `top_k`), `SourceChunk` (`page`, `chunk_id`, `preview`), `AskResponse` Pydantic models |
| `apps/api/schemas/upload.py` | Created | `UploadResponse` schema — `doc_id`, `source`, `n_pages`, `n_chunks` |
| `apps/api/schemas/documents.py` | Created | `DocumentInfo` schema — same fields as `UploadResponse` |
| `apps/api/schemas/__init__.py` | Updated | Re-exports all five schema classes |
| `apps/api/deps.py` | Created | `get_store(request)` and `get_llm(request)` FastAPI dependency functions — read from `app.state`; importable for `dependency_overrides` in tests |
| `apps/api/routers/ask.py` | Created | `POST /ask` — calls `rag_query` directly; maps `list[Chunk]` to `list[SourceChunk]` |
| `apps/api/routers/upload.py` | Created | `POST /upload` — validates `.pdf` extension, calls `chunk_pdf`, adds to store |
| `apps/api/routers/documents.py` | Created | `GET /documents` — aggregates `store._chunks` by `doc_id` into `list[DocumentInfo]` |
| `apps/api/main.py` | Updated | Lifespan creates `Embedder`, `VectorStore`, `LLMClient` on `app.state`; includes three routers; `/health` kept |
| `notebooks/production/…phase05….ipynb` | Updated | Cells [5], [7] replaced with comments; cell [1] header → "active" |
| `.github/workflows/ci.yml` | Updated | Lint targets extended to `apps/`; install step changed to `.[dev,api]` |
| `tests/unit/test_api.py` | Created | 13 tests — see table below |
| `production-ready.md` | Updated | Phase 05 table all rows marked ✅ done |
| `notebook-to-production-steps.md` | Updated | Phase 05 section appended |

### Tests created — `tests/unit/test_api.py` (13 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestHealth` | `test_returns_200` | `GET /health` returns status 200 |
| `TestHealth` | `test_returns_ok` | `GET /health` body is `{"status": "ok"}` |
| `TestAsk` | `test_valid_payload_returns_200` | `POST /ask` with valid payload returns 200 |
| `TestAsk` | `test_response_has_answer_and_sources` | response JSON has `answer` and `sources` keys |
| `TestAsk` | `test_short_question_returns_422` | question shorter than 3 chars returns 422 |
| `TestAsk` | `test_sources_contain_page_and_chunk_id` | each source has `page` and `chunk_id` |
| `TestDocuments` | `test_returns_200` | `GET /documents` returns 200 |
| `TestDocuments` | `test_returns_list` | response is a `list` |
| `TestDocuments` | `test_returns_document_info_shape` | each item has `doc_id`, `source`, `n_pages`, `n_chunks` |
| `TestDocuments` | `test_aggregates_chunks_by_doc_id` | two chunks with same `doc_id` aggregate to one `DocumentInfo` |
| `TestUpload` | `test_pdf_upload_returns_200` | `POST /upload` with a PDF returns 200 |
| `TestUpload` | `test_pdf_upload_returns_expected_fields` | response has `doc_id`, `n_pages`, `n_chunks` with correct chunk count |
| `TestUpload` | `test_non_pdf_returns_400` | non-PDF file returns 400 |

---

## [Phase 04] — End-to-End RAG Pipeline — 2026-06-05

**Commit:** `64f58e4`
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase04-rag-pipeline.ipynb`
**Production notebook:** `notebooks/production/2026-05-25-phase04-rag-pipeline.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/core/llm.py` | Created | `LLMClient` wrapping Ollama; `chat(messages: list[dict], temperature) -> str`; returns plain string (not a dict) |
| `src/mrta/core/rag_pipeline.py` | Created | `rag_query(question, vector_store, llm, top_k) -> dict`; returns `{"answer", "sources": list[Chunk], "latency_s"}` |
| `src/mrta/prompts/__init__.py` | Created | `load_prompt(name, **kwargs)` using Jinja2 `PackageLoader`; renders `src/mrta/prompts/{name}.j2` |
| `src/mrta/prompts/rag.j2` | Created | RAG prompt template — system role, grounding rule, context block with chunk source/page/text, question, citation format |
| `src/mrta/observability/logging.py` | Created | `StructuredLogger.log_run()` appends one JSON line to `settings.log_file`; creates parent directories automatically |
| `src/mrta/observability/__init__.py` | Updated | Exports `StructuredLogger` |
| `src/mrta/__init__.py` | Updated | `LLMClient`, `rag_query`, `load_prompt`, `StructuredLogger` added to public API |
| `notebooks/production/…phase04….ipynb` | Updated | Cells [4], [6], [8], [10], [16] replaced with library imports; cell [1] header → "active" |
| `tests/unit/test_rag_pipeline.py` | Created | 12 tests — see table below |
| `production-ready.md` | Updated | Phase 04 table all rows marked ✅ done; library map updated |
| `notebook-to-production-steps.md` | Updated | Phase 04 section appended |

### Tests created — `tests/unit/test_rag_pipeline.py` (12 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestLLMClient` | `test_chat_returns_mocked_text` | `LLMClient.chat()` returns the mocked response content string |
| `TestLLMClient` | `test_chat_returns_str` | return type is `str` |
| `TestRagQuery` | `test_returns_expected_keys` | result dict has `answer`, `sources`, `latency_s` keys |
| `TestRagQuery` | `test_answer_is_str` | `result["answer"]` is `str` |
| `TestRagQuery` | `test_sources_are_chunk_instances` | every source is a `Chunk` instance |
| `TestRagQuery` | `test_top_k_1_returns_one_source` | `top_k=1` returns exactly 1 source |
| `TestRagQuery` | `test_latency_is_non_negative_float` | `latency_s >= 0` and is `float` |
| `TestLoadPrompt` | `test_returns_nonempty_string` | `load_prompt("rag", ...)` returns non-empty string |
| `TestLoadPrompt` | `test_contains_question` | rendered prompt contains the question text |
| `TestLoadPrompt` | `test_renders_chunk_content` | rendered prompt includes chunk text |
| `TestStructuredLogger` | `test_appends_one_line` | `log_run()` appends exactly one line to the log file |
| `TestStructuredLogger` | `test_json_contains_question_and_answer` | logged JSON has `question` and `answer` keys |

---

## [Phase 03] — Embeddings & FAISS — 2026-06-05

**Commit:** pending  
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase03-embeddings-and-faiss.ipynb`  
**Production notebook:** `notebooks/production/2026-05-25-phase03-embeddings-and-faiss.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/retrieval/embedder.py` | Created | New `Embedder` wrapper class — not inline in tutorial; routes to sentence-transformers (`"/" in model_name`) or Ollama REST API |
| `src/mrta/retrieval/vector_store.py` | Created | `VectorStore` extracted from tutorial cell [14] and refined — `__init__(embedder)` drops explicit `dim`; `search()` returns `list[Chunk]` not `list[dict]` |
| `src/mrta/retrieval/__init__.py` | Updated | Exports `Embedder`, `VectorStore` |
| `src/mrta/__init__.py` | Updated | `Embedder`, `VectorStore` added to top-level public API |
| `notebooks/production/…phase03….ipynb` | Updated | Cells [4], [6], [14], [18] switched to `from mrta.*` imports; cells [8], [10], [12], [16] kept inline as teaching demos |
| `tests/unit/test_vector_store.py` | Created | 12 tests — see table below |
| `production-ready.md` | Updated | `retrieval/embedder.py`, `retrieval/vector_store.py` → `✅ done` |
| `notebook-to-production-steps.md` | Updated | Phase 03 section appended |

### Tests created — `tests/unit/test_vector_store.py` (12 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestEmbedder` | `test_embed_returns_correct_shape` | `embed(["hello"])` shape is `(1, dim)` |
| `TestEmbedder` | `test_embed_float32` | output dtype is `float32` |
| `TestEmbedder` | `test_embed_normalised` | L2 norms of all rows are `~1.0` |
| `TestEmbedder` | `test_dim_positive` | `embedder.dim > 0` |
| `TestEmbedder` | `test_model_name_accessible` | `embedder.model_name != ""` |
| `TestVectorStore` | `test_search_returns_k_results` | `search(q, k=1)` returns exactly 1 result |
| `TestVectorStore` | `test_search_returns_chunk_instances` | every result is a `Chunk` instance |
| `TestVectorStore` | `test_search_chunks_have_text` | every result has non-empty `.text` |
| `TestVectorStore` | `test_search_k_capped_at_index_size` | `k > n_chunks` returns `n_chunks`, not `k` |
| `TestVectorStore` | `test_save_load_roundtrip` | reloaded store returns same top chunk for same query |
| `TestVectorStore` | `test_save_creates_expected_files` | `save()` writes `index.faiss`, `metadata.jsonl`, `config.json` |
| `TestVectorStore` | `test_add_empty_is_noop` | `add([])` does not raise; `search()` returns `[]` |

---

## [Phase 02] — Chunking Strategies — 2026-06-05

**Commit:** `836e69a`  
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase02-chunking-strategies.ipynb`  
**Production notebook:** `notebooks/production/2026-05-25-phase02-chunking-strategies.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/core/schemas.py` | Updated | `Chunk` model added — `chunk_id`, `doc_id`, `source`, `page`, `text`, `section?`, `n_tokens?` |
| `src/mrta/ingestion/chunker.py` | Created | `fixed_chunks`, `recursive_chunks`, `token_chunks`, `semantic_chunks`, `chunk_pdf` dispatcher; optional deps imported lazily inside each function body |
| `src/mrta/ingestion/__init__.py` | Updated | Exports `chunk_pdf` alongside `load_pdf` |
| `src/mrta/__init__.py` | Updated | `Chunk`, `chunk_pdf` added to top-level public API |
| `notebooks/production/…phase02….ipynb` | Updated | Cells [4], [6], [8], [10], [12], [17] switched to `from mrta.*` imports; cell [14] (`summarize`) kept inline |
| `tests/unit/test_chunker.py` | Created | 14 tests — see table below |
| `production-ready.md` | Updated | `ingestion/chunker.py` → `✅ done`; `core/schemas.py` updated to note `Chunk` added |
| `notebook-to-production-steps.md` | Updated | Phase 02 section appended |

### Tests created — `tests/unit/test_chunker.py` (14 tests)

| Test class | Test | Assertion |
|------------|------|-----------|
| `TestFixedChunks` | `test_returns_chunks` | `len(chunks) > 0` |
| `TestFixedChunks` | `test_all_are_chunk_instances` | every item is a `Chunk` |
| `TestFixedChunks` | `test_chunk_ids_unique` | no duplicate `chunk_id` values |
| `TestFixedChunks` | `test_all_chunks_have_required_fields` | `doc_id`, `source`, `page ≥ 1`, `text.strip()` non-empty |
| `TestFixedChunks` | `test_chunk_id_format` | `_p{page}_c` present in every `chunk_id` |
| `TestFixedChunks` | `test_doc_id_matches_pdf` | chunk `doc_id` == `pdf.doc_id` |
| `TestTokenChunks` | `test_returns_chunks` | `len > 0` (skips if `tiktoken` absent) |
| `TestTokenChunks` | `test_n_tokens_within_size` | every `n_tokens ≤ size` (skips if `tiktoken` absent) |
| `TestTokenChunks` | `test_chunk_ids_unique` | no duplicate IDs (skips if `tiktoken` absent) |
| `TestRecursiveChunks` | `test_returns_chunks` | `len > 0` (skips if `langchain_text_splitters` absent) |
| `TestRecursiveChunks` | `test_chunk_ids_unique` | no duplicate IDs (skips if `langchain_text_splitters` absent) |
| `TestChunkPdf` | `test_default_strategy_is_recursive` | `chunk_pdf(pdf)` returns chunks (default strategy) |
| `TestChunkPdf` | `test_fixed_strategy` | `chunk_pdf(pdf, strategy="fixed")` returns chunks |
| `TestChunkPdf` | `test_unknown_strategy_raises` | `strategy="nonexistent"` raises `ValueError` |

---

## [Phase 01] — PDF Ingestion — 2026-06-05

**Commit:** `013a47c`  
**Tutorial notebook:** `notebooks/tutorials/2026-05-25-phase01-pdf-ingestion.ipynb`  
**Production notebook:** `notebooks/production/2026-05-25-phase01-pdf-ingestion.ipynb`

### Changed files

| File | Change | Notes |
|------|--------|-------|
| `src/mrta/ingestion/pdf_loader.py` | Updated | `ocr_page_if_needed(page, dpi=200)` extracted as a public function — text fallback or pytesseract OCR for scanned pages |
| `notebooks/production/…phase01….ipynb` | Updated | Cell [18] activated — `from mrta.ingestion.pdf_loader import ocr_page_if_needed` |
| `production-ready.md` | Created | Architecture blueprint — per-phase extraction plan, library map, interface signatures |
| `notebook-to-production-steps.md` | Created | Execution log — Phase 00 and Phase 01 sections; SHA-1 stable ID concept note |

### Tests

`tests/unit/test_ingestion.py` (14 tests) was created in an earlier phase (commit `5db2172`) and
covers `load_pdf` and `_doc_id`. No new test file was added in Phase 01 — `ocr_page_if_needed`
requires a live Ollama or scanned PDF and is tested manually via the production notebook.

---

## Earlier phases — Library scaffold (2026-06-04)

These commits established the package structure before notebook-to-production work began.

| Commit | Description |
|--------|-------------|
| `5db2172` | `tests/unit/test_ingestion.py` — 14 tests for `load_pdf`, `_doc_id` |
| `af53763` | `.github/workflows/ci.yml` — ruff + black + pytest on push/PR |
| `0a5e527` | `docs/adr/` — ADR-001 through ADR-006 |
| `486134e` | `scripts/ingest.py`, `docker/Dockerfile.api`, `docker/docker-compose.yml` |
| `40ac6f5` | `notebooks/production/` — 10 production notebook shells |
| `5046f20` | `src/mrta/__init__.py` public API, VLM model aligned with ADR-003 |
| `34f0070` | `src/mrta/core/config.py` — YAML-backed settings with `MRTA_ENV` switching |
| `3f11477` | `src/mrta/` library layout, `pyproject.toml`, optional dep groups |
