# Multimodal RAG Implementation Plan

Status legend: ✅ Complete | 🔄 In progress | ⬜ Not started

---

# Stage 1 — Visual Evidence Foundation ✅

Branch: `feature/mmrag-visual-evidence` → merged to `main`

---

## 1. Existing Relevant Architecture

### Schemas (`src/mrta/core/schemas.py`)

- `FigureRecord` — `doc_id`, `source`, `page`, `figure_index`, `image_bytes`, `to_pil()`
  - Missing: `width`, `height`, `bbox`, `nearby_text`
- `Chunk` — text evidence, already has `chunk_id`, `source`, `page`
- No `EvidenceRecord` exists yet

### Ingestion

- `figure_extractor.py` — raster extraction via PyMuPDF; returns `list[FigureRecord]`
  - Captures embedded raster images only; skips vector graphics
  - No dimensions, no bbox, no nearby text
- No `page_renderer.py` exists

### VLM client (`src/mrta/multimodal/vlm_client.py`)

- `VLMClient.caption(image, prompt)` — single-image only, no general `generate()` API
- Uses `ollama.chat` internally

### CLIP (`src/mrta/multimodal/clip_embedder.py`)

- `CLIPEmbedder.embed_image(image)` → `(512,) float32`
- `CLIPEmbedder.embed_text(text)` → `(512,) float32`
- Both L2-normalised → dot-product = cosine similarity

### Config (`src/mrta/core/config.py`)

- Has `ollama_vlm_model`, `clip_model`
- Missing: `page_render_dpi` and other visual settings

### Testing pattern

- `pytest.importorskip()` for heavy deps (PyMuPDF, open_clip)
- `unittest.mock.patch` + `MagicMock` for Ollama calls
- `tests/fixtures/sample.pdf` used by existing figure extractor tests

---

## 2. Files That Will Change

| File | Change |
|---|---|
| `src/mrta/core/schemas.py` | Add `EvidenceRecord`; extend `FigureRecord` with `width`, `height`, `bbox`, `nearby_text` |
| `src/mrta/core/config.py` | Add `page_render_dpi: int = 150` |
| `src/mrta/ingestion/figure_extractor.py` | Populate new `FigureRecord` fields; add `to_evidence_record()` |
| `src/mrta/multimodal/vlm_client.py` | Add `generate(prompt, images)`; refactor `caption()` to call it |
| `src/mrta/__init__.py` | Expose `EvidenceRecord`, `PageRenderer`, `VisualAnalyzer`, `VisualDescription` |

---

## 3. Files That Will Be Added

| File | Purpose |
|---|---|
| `src/mrta/ingestion/page_renderer.py` | Render PDF pages as PIL images at configurable DPI → `EvidenceRecord(modality="page")` |
| `src/mrta/multimodal/visual_analyzer.py` | Structured VLM-based description of a figure; `VisualDescription` schema + text serialization |
| `tests/unit/test_evidence_record.py` | EvidenceRecord schema, field validation, serialization |
| `tests/unit/test_page_renderer.py` | Page rendering (uses sample.pdf fixture, no Ollama) |
| `tests/unit/test_vlm_client.py` | VLMClient.generate() with mocked ollama |
| `tests/unit/test_visual_analyzer.py` | VisualAnalyzer with mocked VLMClient |

---

## 4. Compatibility Considerations

- `FigureRecord` keeps all existing fields; new fields are `Optional` with defaults
- `extract_figures()` signature unchanged; returns richer records
- `VLMClient.caption()` signature unchanged; delegates to new `generate()`
- All existing tests must pass without modification
- No new mandatory dependencies (page rendering uses `fitz` which is already in `[pdf]`)

---

## 5. Tests to Add

- `test_evidence_record.py`: construction, `modality` enum, round-trip JSON, `FigureRecord.to_evidence_record()`
- `test_page_renderer.py`: `render_page()` shape, dtype, DPI scaling; `render_pages()` count; skip if `fitz` absent
- `test_vlm_client.py`: `generate()` with mocked `ollama.chat`; `caption()` still works via `generate()`; `is_available()` mock
- `test_visual_analyzer.py`: `VisualDescription` construction; `to_retrieval_text()` format; `VisualAnalyzer.analyze()` with mocked VLM

---

## 6. Notebook Changes

None in Stage 1 — notebooks are addressed in Stage 2 (Phase 07) and beyond.

---

## 7. Implementation Order

1. Extend `FigureRecord` + add `EvidenceRecord` in `schemas.py`
2. Update `figure_extractor.py` to populate new fields
3. Add `page_renderer.py`
4. Generalize `VLMClient.generate()`
5. Add `visual_analyzer.py`
6. Add `page_render_dpi` to `config.py`
7. Write all tests
8. Update `__init__.py` exports
9. Run quality gates: `pytest`, `ruff`, `black`, `mypy`

## Outcome

All items implemented and merged. 236 tests passing.

---

# Stage 2 — Caption-based Visual Retrieval ✅

Branch: `feature/mmrag-caption-retrieval` → merged to `main`

## 1. Existing Relevant Architecture

- `EvidenceRecord` — available from Stage 1; carries `retrieval_text()` fallback chain
- `Embedder` — sentence-transformers wrapper, `embed(texts) → (n, 384) float32`
- `VectorStore` — FAISS `IndexFlatIP` for text `Chunk` objects
- `VisualAnalyzer` / `VisualDescription` — available from Stage 1
- No caption-specific retrieval store existed yet

## 2. Files Changed

| File | Change |
|---|---|
| `src/mrta/__init__.py` | Add `CaptionVectorStore` to retrieval exports |
| `tests/unit/test_vector_store.py` | Hardcode `Embedder("sentence-transformers/all-MiniLM-L6-v2")` in `real_embedder` fixture; remove stale Ollama `skipif` |

## 3. Files Added

| File | Purpose |
|---|---|
| `src/mrta/retrieval/caption_store.py` | FAISS `IndexFlatIP` store for `EvidenceRecord` objects indexed by `retrieval_text()` using sentence-transformers |
| `tests/unit/test_caption_store.py` | 17 tests — mocked `Embedder`, deterministic orthogonal unit vectors, covers add/search/immutability/metadata |

## 4. Key Design Decisions

- `CaptionVectorStore` kept separate from `VectorStore`: `EvidenceRecord` carries modality metadata `Chunk` does not; return types differ
- Embedding space: sentence-transformer (384-dim), same model as text chunks — scores are directly comparable
- Immutability: `search()` returns `record.model_copy(update={"retrieval_score": score})`; stored records never mutated
- Embedder fixture: hardcoded model ID in tests to avoid config singleton / Ollama dependency ordering issue

## 5. Notebook Changes

- `notebooks/production/2026-05-25-phase07-figure-extraction-and-vlm.ipynb` — full rewrite (25 cells): sections 7.1–7.5 with `CaptionVectorStore` pipeline
- `notebooks/tutorials/2026-05-25-phase07-figure-extraction-and-vlm.ipynb` — full rewrite (24 cells): all implementation inline (`SimpleCaptionStore`, `analyze_figure()` etc.); no `mrta` imports in implementation cells

## 6. Implementation Order

1. `caption_store.py` — core implementation
2. `test_caption_store.py` — 17 unit tests
3. Export in `__init__.py`
4. Fix `real_embedder` fixture in `test_vector_store.py`
5. Rewrite both Phase 07 notebooks
6. Quality gates: 236 tests passing, ruff + black clean

## Outcome

> Figure information can be retrieved through VLM-generated descriptions using the existing text retrieval infrastructure.

---

# Stage 3 — CLIP Visual Retrieval ✅

Branch: `feature/mmrag-clip-retrieval` → merged to `main`

## 1. Existing Relevant Architecture

- `CLIPEmbedder` (`src/mrta/multimodal/clip_embedder.py`) — already implemented
  - `embed_image(PIL.Image) → (512,) float32` L2-normalised
  - `embed_text(str) → (512,) float32` L2-normalised
  - Dot-product is cosine similarity across modalities (shared CLIP space)
- `EvidenceRecord` — carries `image_bytes`; `.to_pil()` reconstructs PIL image
- `CaptionVectorStore` — sentence-transformer space (384-dim); **not** the store being built here
- `VectorStore` — sentence-transformer space (384-dim); also separate

## 2. Files That Will Change

| File | Change |
|---|---|
| `src/mrta/__init__.py` | Add `VisualVectorStore` to multimodal try-block and `__all__` |

## 3. Files to Add

| File | Purpose |
|---|---|
| `src/mrta/retrieval/visual_vector_store.py` | FAISS `IndexFlatIP` (512-dim) store for `EvidenceRecord`; images embedded via `CLIPEmbedder.embed_image()`, queries via `embed_text()` |
| `tests/unit/test_visual_vector_store.py` | Unit tests with mocked `CLIPEmbedder`; covers add/skip/search/immutability/metadata/persistence |

## 4. Key Design Decisions

| Concern | Decision |
|---|---|
| Embedding space | CLIP (512-dim) — completely separate from sentence-transformer stores |
| Records without image bytes | Skipped silently on `add()` (vector-graphic page records may lack bytes at index time) |
| Query embedding | `embed_text()` on `CLIPEmbedder` — NOT the sentence-transformer `Embedder` |
| Image embedding | `embed_image(record.to_pil())` for each record on `add()` |
| Persistence | `save(path)` / `load(path, embedder)` consistent with `VectorStore` |

## 5. Interface

```python
store = VisualVectorStore(clip_embedder)
store.add(evidence_records)                      # embeds images via CLIP
results = store.search(query, k=5)               # text → image, returns list[EvidenceRecord]
pairs   = store.search_with_scores(query, k=5)   # list[tuple[EvidenceRecord, float]]
store.size                                        # int
store.save(path)
store = VisualVectorStore.load(path, clip_embedder)
```

## 6. Tests to Add

- Empty store → `size == 0`, `search()` returns `[]`
- `add()` skips records with no `image_bytes`
- `add()` calls `embed_image()`, not `embed_text()`
- `search()` returns `EvidenceRecord` instances
- Top-1 returns closest match (deterministic orthogonal unit vectors)
- `retrieval_score` set on returned records
- Stored records not mutated by `search()`
- k > size returns all records
- Modality and page metadata preserved on results
- `save()` / `load()` round-trip preserves top-1 result

## 7. Implementation Order

1. `visual_vector_store.py` — core implementation
2. `test_visual_vector_store.py` — unit tests
3. Export in `__init__.py`
4. Quality gates: `pytest`, `ruff`, `black`

## Expected Outcome

> A natural-language query can retrieve the original relevant figure without relying on any VLM-generated description.

## Outcome

256 → 276 tests passing. `VisualVectorStore` fully implemented and merged.

---

# Stage 4 — Multimodal Fusion 🔄

Branch: `feature/mmrag-fusion` → ready to merge

Covers sections 9, 10, 11 + section 20 (Phase 07b notebook) from `notes/Multimodal RAG implementation.md`.

## 1. Existing Relevant Architecture

- `VectorStore` — FAISS `IndexFlatIP` for `Chunk` objects (sentence-transformer, 384-dim)
- `CaptionVectorStore` — FAISS `IndexFlatIP` for `EvidenceRecord` via `retrieval_text()` (sentence-transformer, 384-dim)
- `VisualVectorStore` — FAISS `IndexFlatIP` for `EvidenceRecord` via CLIP image embeddings (CLIP, 512-dim)
- `EvidenceRecord.from_chunk()` — converts a `Chunk` to a text-modality `EvidenceRecord`, already implemented in `schemas.py`
- `Reranker` — cross-encoder reranker operating on `Chunk.text` (sentence-transformers CrossEncoder)
- Three stores are in independent embedding spaces; fusion must happen at the **ranking level**, not by averaging scores

## 2. Files to Add

| File | Purpose |
|---|---|
| `src/mrta/retrieval/fusion.py` | `FusedResult` dataclass + `reciprocal_rank_fusion()` pure function |
| `src/mrta/retrieval/multimodal_retriever.py` | `MultimodalRetriever` combining text/caption/visual retrieval and RRF; optional reranking hook |
| `tests/unit/test_fusion.py` | ~15 unit tests for RRF (pure function, no model deps) |
| `tests/unit/test_multimodal_retriever.py` | ~15 unit tests for `MultimodalRetriever` (all stores mocked) |
| `notebooks/production/2026-08-25-phase07b-multimodal-retrieval.ipynb` | Production notebook: sections 07b.1–07b.7 |
| `notebooks/tutorials/2026-08-25-phase07b-multimodal-retrieval.ipynb` | Tutorial notebook: same sections, implementation inline |

## 3. Files to Modify

| File | Change |
|---|---|
| `src/mrta/__init__.py` | Add `MultimodalRetriever`, `FusedResult`, `reciprocal_rank_fusion` to retrieval exports |
| `notes/multimodal_rag_implementation_plan.md` | Stage 3 → ✅, this Stage 4 section |

## 4. Key Interfaces

### `fusion.py`

```python
@dataclass
class FusedResult:
    record: EvidenceRecord        # deduplicated record (highest-ranked version kept)
    rrf_score: float              # Σ 1/(k + rank_r(d)) across all lists
    per_list_rank: dict[str, int] # list_name → 1-based rank (absent = not in that list)
    source_modality: str          # record.modality ("text" | "image" | "page")

def reciprocal_rank_fusion(
    named_lists: dict[str, list[EvidenceRecord]],
    k: int = 60,
    top_n: int | None = None,
) -> list[FusedResult]:
    """Fuse any number of named ranked lists. Deduplicates by evidence_id."""
```

### `multimodal_retriever.py`

```python
class MultimodalRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        caption_store: CaptionVectorStore | None = None,
        visual_store: VisualVectorStore | None = None,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        reranker_top_n: int = 5,
    ) -> None: ...

    def retrieve(
        self,
        query: str,
        k_text: int = 5,
        k_visual: int = 5,
        k_final: int = 8,
    ) -> list[EvidenceRecord]:
        """Returns top k_final EvidenceRecords with retrieval_score = RRF score."""

    def retrieve_with_fusion_details(
        self,
        query: str,
        k_text: int = 5,
        k_visual: int = 5,
        k_final: int = 8,
    ) -> list[FusedResult]:
        """Same as retrieve() but returns full FusedResult list for diagnostics."""
```

## 5. Design Decisions

| Concern | Decision |
|---|---|
| RRF formula | `Σ 1/(k + rank_r(d))` with 1-based ranks; missing from a list contributes 0 |
| Deduplication | By `evidence_id`; when a record appears in multiple lists, the highest-ranked (lowest rank number) copy is retained as the canonical record |
| Score incompatibility | Raw cosine scores from sentence-transformer and CLIP are never compared; only rank order feeds RRF |
| Optional stores | `MultimodalRetriever` works with only `VectorStore` (text-only fallback) |
| Reranking limitation | Cross-encoder operates on `EvidenceRecord.retrieval_text()` — visual evidence quality limited to caption/description text; clearly documented |
| `named_lists` key convention | `"text"`, `"caption"`, `"visual"` — consistent naming across retriever and tests |

## 6. Tests to Add

**`test_fusion.py`**

- Empty lists → empty result
- Single list passthrough
- Two lists, same doc ranks top
- RRF score formula verification (`1/(60+1) + 1/(60+2) ≈ expected`)
- Deduplication by evidence_id (same record in two lists → one result)
- Canonical record is highest-ranked instance
- `per_list_rank` keys present for each list name
- Absent from list → key absent in `per_list_rank`
- `top_n` limits results
- `k` configurable
- All lists empty → empty
- Order: highest RRF score first
- Three lists fused correctly

**`test_multimodal_retriever.py`**

- Text-only (no caption/visual store) → text results
- With caption store → fused text+caption
- With visual store → fused text+visual
- All three stores → fused triple
- `retrieve()` returns `EvidenceRecord` with `retrieval_score` = RRF score
- `retrieve_with_fusion_details()` returns `FusedResult` list
- `k_final` limits output
- Empty visual store → still returns text results
- Optional reranker applied to top candidates
- Reranker not called when `reranker=None`
- `per_list_rank` populated correctly in fusion details

## 7. Implementation Order

1. Update `multimodal_rag_implementation_plan.md` (this document)
2. Implement `fusion.py` — isolated pure function
3. Write `test_fusion.py` — run: `pytest tests/unit/test_fusion.py -v`
4. Implement `multimodal_retriever.py`
5. Write `test_multimodal_retriever.py` — run: `pytest tests/unit/test_multimodal_retriever.py -v`
6. Update `__init__.py` exports
7. Run full quality gates: `pytest`, `ruff`, `black`
8. Create production and tutorial Phase 07b notebooks

## Expected Outcome

> Text and visual retrieval run independently and their rankings are fused cleanly via RRF.
> A single `MultimodalRetriever.retrieve()` call returns the best mixed-modality evidence for any query.

## Outcome

292 tests passing. `fusion.py`, `multimodal_retriever.py`, 36 new tests, and both
Phase 07b notebooks implemented and merged.

---

# Stage 5 — Full Multimodal RAG 🔄

Branch: `feature/mmrag-generation`

Covers spec sections 12 (generation), 13 (grounded prompt), 14 (structured citations),
15 (fallback), 16 (config) + section 21 (Phase 07c notebook).

---

## 1. Existing Relevant Architecture

### Generation

- `rag_pipeline.rag_query()` — text-only RAG: retrieve `Chunk`s → `load_prompt("rag")` → `LLMClient.chat()` → returns `dict`
- `VLMClient.generate(prompt, images)` — already generalised in Stage 1;
  accepts `Sequence[PIL.Image]`; raises `LLMError` on model not found
- `load_prompt(name, **kwargs)` — Jinja2 template renderer; templates live in `src/mrta/prompts/`
- `rag.j2` — text-only template with `[chunk N | source | page]` labelling

### Retrieval

- `MultimodalRetriever.retrieve(query, k_text, k_visual, k_final)` → `list[EvidenceRecord]`
  - Each record has `modality` in `{"text", "image", "page"}`, `image_bytes` (may be `None`),
    `text`, `caption`, `detailed_description`, `page`, `figure_index`, `source`
  - Records without visual stores degrade automatically to text-only
  - `retrieval_score` set to RRF score

### Schemas

- `EvidenceRecord` — carries all evidence; `retrieval_text()` returns best textual
  representation; `to_pil()` reconstructs image from `image_bytes`
- `EvalReport` — aggregated eval results (existing, unrelated)
- No `MultimodalCitation` or `MultimodalAnswer` exist yet

### Config

- `Settings`: `ollama_vlm_model`, `clip_model`, `top_k`, `page_render_dpi`
- Missing: `enable_multimodal_rag`, `visual_top_k`, `text_top_k`, `fusion_top_k`, `rrf_k`

---

## 2. Files That Will Change

| File | Change |
|---|---|
| `src/mrta/core/schemas.py` | Add `MultimodalCitation`, `MultimodalAnswer` Pydantic models |
| `src/mrta/core/config.py` | Add `enable_multimodal_rag`, `visual_top_k`, `text_top_k`, `fusion_top_k`, `rrf_k` |
| `src/mrta/generation/__init__.py` | Export `MultimodalRAG` |
| `src/mrta/__init__.py` | Add `MultimodalRAG`, `MultimodalCitation`, `MultimodalAnswer` to top-level exports |

---

## 3. Files to Add

| File | Purpose |
|---|---|
| `src/mrta/generation/multimodal_rag.py` | `MultimodalRAG` class — wires retriever + VLM; builds prompt; returns `MultimodalAnswer` |
| `src/mrta/prompts/multimodal_rag.j2` | Grounded multimodal prompt: `[T1]…[Tn]` text evidence + `[V1]…[Vn]` visual evidence with citation instructions |
| `tests/unit/test_multimodal_rag.py` | ~18 unit tests (all mocked — no Ollama, no CLIP in CI) |
| `notebooks/production/2026-08-26-phase07c-multimodal-rag.ipynb` | Production notebook — 07c.1–07c.7, `mrta` imports throughout |
| `notebooks/tutorials/2026-08-26-phase07c-multimodal-rag.ipynb` | Tutorial notebook — same sections, inline implementations cross-validated against production |

---

## 4. New Schemas (`schemas.py`)

```python
class MultimodalCitation(BaseModel):
    label: str                               # "[T1]" or "[V1]"
    evidence_id: str
    modality: Literal["text", "image", "page"]
    source: str
    page: int
    figure_index: int | None = None

class MultimodalAnswer(BaseModel):
    answer: str
    text_citations: list[MultimodalCitation]
    visual_citations: list[MultimodalCitation]
    retrieval_mode: Literal["multimodal", "text_only"] = "multimodal"
    latency_s: float
```

---

## 5. `MultimodalRAG` Interface (`generation/multimodal_rag.py`)

```python
class MultimodalRAG:
    def __init__(
        self,
        retriever: MultimodalRetriever,
        vlm: VLMClient,
        text_top_k: int = 5,
        visual_top_k: int = 5,
        fusion_top_k: int = 8,
    ) -> None: ...

    def ask(
        self,
        question: str,
        source_filter: str | None = None,
    ) -> MultimodalAnswer:
        """Full multimodal RAG: retrieve → fuse → prompt+images → VLM → grounded answer.

        Fallback: if VLM raises LLMError, degrades to text-only answer using
        retrieved text evidence, with retrieval_mode="text_only".
        """

    def _split_evidence(
        self, evidence: list[EvidenceRecord]
    ) -> tuple[list[EvidenceRecord], list[EvidenceRecord]]:
        """Split fused evidence into text and visual lists."""

    def _build_prompt_and_images(
        self,
        question: str,
        text_ev: list[EvidenceRecord],
        visual_ev: list[EvidenceRecord],
    ) -> tuple[str, list[PIL.Image]]:
        """Render multimodal_rag.j2; collect PIL images from visual records with bytes."""

    def _make_citations(
        self,
        text_ev: list[EvidenceRecord],
        visual_ev: list[EvidenceRecord],
    ) -> tuple[list[MultimodalCitation], list[MultimodalCitation]]:
        """Build [T1]…[Tn] and [V1]…[Vn] citation lists."""
```

---

## 6. Prompt Template (`prompts/multimodal_rag.j2`)

Conceptual structure (Jinja2):

```jinja2
You are a research assistant. Answer using ONLY the evidence below.
Cite text evidence as [T1], [T2], ... and visual evidence as [V1], [V2], ...
When referring to a figure, include page and figure number.
If evidence is insufficient, say so explicitly.

--- QUESTION ---
{{ question }}

--- TEXT EVIDENCE ---
{% for ev in text_evidence %}
[T{{ loop.index }}]
Document: {{ ev.source }} | Page: {{ ev.page }}
{{ ev.text }}

{% endfor %}
--- VISUAL EVIDENCE ---
{% for ev in visual_evidence %}
[V{{ loop.index }}]
Document: {{ ev.source }} | Page: {{ ev.page }}{% if ev.figure_index %} | Figure: {{ ev.figure_index }}{% endif %}
{% if ev.caption or ev.detailed_description %}Description: {{ ev.caption or ev.detailed_description }}{% endif %}
<image attached>

{% endfor %}
--- ANSWER (cite [T#] and [V#]) ---
```

Images with `image_bytes=None` (vector-graphic page records) are listed in the prompt
text only — no PIL image attached to the VLM call.

---

## 7. Fallback Hierarchy (spec section 15)

```text
Full multimodal RAG
      ↓ VLMClient raises LLMError (model unavailable)
Text answer using retrieved text evidence (retrieval_mode="text_only")
      ↓ No visual store / empty visual index
Same code path — visual_evidence list is empty; VLM sees only text chunks
```

The fallback is handled inside `ask()` with a `try/except LLMError` block.
No silent swallowing — `retrieval_mode` field communicates what actually ran.

---

## 8. Config Additions (`config.py`)

```python
# Multimodal RAG
enable_multimodal_rag: bool = True
visual_top_k: int = 5
text_top_k: int = 5
fusion_top_k: int = 8
rrf_k: int = 60
```

Existing `top_k` is kept for backward-compatible text-only RAG; the new fields govern the multimodal path.

---

## 9. Tests (`test_multimodal_rag.py`)

All stores and VLM mocked — no Ollama, no CLIP, no FAISS in CI.

| Test | Verifies |
|---|---|
| `test_ask_text_and_visual` | Full path returns `MultimodalAnswer` |
| `test_ask_text_only_when_no_visual_evidence` | VLM called, visual citations empty |
| `test_ask_vlm_unavailable_falls_back` | `LLMError` → `retrieval_mode="text_only"`, VLM not called with images |
| `test_retrieval_mode_multimodal_on_success` | `retrieval_mode="multimodal"` set |
| `test_retrieval_mode_text_only_on_fallback` | `retrieval_mode="text_only"` on `LLMError` |
| `test_text_citation_labels` | `[T1]`, `[T2]` assigned correctly |
| `test_visual_citation_labels` | `[V1]`, `[V2]` assigned correctly |
| `test_citation_source_page_figure` | `source`, `page`, `figure_index` populated |
| `test_images_with_bytes_attached` | Records with `image_bytes` yield PIL images to VLM |
| `test_images_without_bytes_excluded` | Records with `image_bytes=None` not attached |
| `test_vlm_receives_original_images` | `vlm.generate()` receives PIL images, not caption strings |
| `test_latency_s_populated` | `latency_s > 0` |
| `test_source_filter_passed` | `source_filter` forwarded to retriever |
| `test_fusion_top_k_limits_evidence` | `fusion_top_k` caps total evidence passed to prompt |
| `test_empty_retrieval_returns_answer` | Zero evidence → VLM still called, answer returned |
| `test_text_citations_have_correct_modality` | All `modality="text"` |
| `test_visual_citations_have_correct_modality` | `modality` in `{"image", "page"}` |
| `test_multiple_visual_images_passed` | Multiple PIL images in a single VLM call |

---

## 10. Phase 07c Notebook Sections (spec section 21)

**Main question:** How do we reason over retrieved text and images together?

| Section | Content |
|---|---|
| 07c.1 Retrieve text | Normal text retrieval baseline |
| 07c.2 Retrieve visual evidence | CLIP visual retrieval; display retrieved figures |
| 07c.3 Fuse evidence | `MultimodalRetriever`; inspect modality, page, RRF score |
| 07c.4 Construct multimodal context | Build prompt manually (tutorial); explain text vs VLM evidence split |
| 07c.5 Multimodal VLM generation | Send question + text + original images to VLM |
| 07c.6 Grounded answer | Resolve `[T1]`/`[V1]` → document/page/figure |
| 07c.7 Compare four systems | Text RAG vs Caption RAG vs CLIP+text vs Full Multimodal RAG |

Tutorial notebook defines `build_multimodal_prompt()`, `multimodal_ask()` inline,
then cross-validates output against production `mrta.MultimodalRAG`.

---

## 11. Compatibility Considerations

- `rag_query()` and `rag_pipeline.py` untouched — text-only RAG unchanged
- `EvidenceRecord`, `FigureRecord`, `Chunk` unchanged
- New schemas (`MultimodalCitation`, `MultimodalAnswer`) additive only
- New config fields all have defaults — existing `.env` files and YAML configs unaffected
- Existing `mrta.__init__.py` exports preserved

---

## 12. Implementation Order

1. Add `MultimodalCitation` + `MultimodalAnswer` to `schemas.py`
2. Add config fields to `config.py`
3. Add `src/mrta/prompts/multimodal_rag.j2`
4. Implement `src/mrta/generation/multimodal_rag.py`
5. Update `src/mrta/generation/__init__.py`
6. Update `src/mrta/__init__.py`
7. Write `tests/unit/test_multimodal_rag.py`
8. Run quality gates: `pytest`, `ruff`, `black`
9. Create production Phase 07c notebook
10. Create tutorial Phase 07c notebook

---

## Expected Outcome

> Retrieved text and original images are passed to the VLM and the answer cites both forms of evidence.
> If the VLM is unavailable, the system falls back to a text-only answer without crashing.

---

# Stage 6 — Teaching Modes ✅

Covers: Explain, Socratic, Quiz, Compare, Visual Evidence modes; API + Streamlit integration; Phase 08 notebook.

## Outcome (352 tests passing, ruff clean, black clean)

### Architecture

All five teaching modes share the same `MultimodalRetriever` retrieval pipeline.
The mode parameter selects a Jinja2 prompt template — only the rendered prompt changes;
retrieval, image-passing, and citation logic are identical across modes.

### New files

- `src/mrta/prompts/teaching_explain.j2` — explain at undergraduate level
- `src/mrta/prompts/teaching_socratic.j2` — guiding questions, no direct answer
- `src/mrta/prompts/teaching_quiz.j2` — 5 questions + answer key from evidence
- `src/mrta/prompts/teaching_compare.j2` — structural comparison of figures/claims
- `src/mrta/prompts/teaching_visual_evidence.j2` — per-figure relevance analysis
- `tests/unit/test_teaching_modes.py` — 17 tests across 3 classes
- `notebooks/production/2026-08-26-phase08-teaching-modes.ipynb`
- `notebooks/tutorials/2026-08-26-phase08-teaching-modes.ipynb`

### Modified files

- `src/mrta/generation/multimodal_rag.py` — added `teaching_mode: str | None = None` parameter;
  `VALID_TEACHING_MODES` frozenset; `_build_prompt_and_images` dispatches to
  `teaching_{mode}.j2` or `multimodal_rag.j2`
- `src/mrta/prompts/__init__.py` — added 5 teaching modes to `MODES` dict
- `apps/api/schemas/ask.py` — `AskRequest`: `retrieval_mode`, `teaching_mode` fields;
  `AskResponse`: `retrieval_mode`, `visual_sources` fields; new `VisualSource` schema
- `apps/api/deps.py` — `get_retriever()`, `get_vlm()` dependency functions
- `apps/api/main.py` — lifespan initialises multimodal stack
  (`_CLIPEmbedder`, `_VisualVectorStore`, `_MultimodalRetriever`, `_VLMClient`)
  with graceful fallback to `None` when the extra is unavailable
- `apps/api/routers/ask.py` — dispatches on `retrieval_mode`;
  multimodal path uses `MultimodalRAG`; 503 when retriever unavailable;
  text path unchanged
- `apps/streamlit/app.py` — retrieval mode radio (Text / Multimodal);
  teaching mode selectbox for multimodal path; visual evidence expandable section;
  503 error handling
- `tests/unit/test_api.py` — extended fixtures to patch multimodal lifespan imports;
  `TestAskMultimodal` (10 tests)

---

# Stage 7 — Evaluation ⬜

Covers: multimodal eval dataset, Figure Recall@k, Multimodal Recall@k, citation correctness, Phase 09 notebook.

---

# Stage 8 — Production Polish ⬜

Covers: README, architecture docs, `.env.example`, Docker docs, CHANGELOG, ADR updates.
