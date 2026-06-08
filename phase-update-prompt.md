# Phase Update Prompt

Paste this at the start of any notebook-to-production session, replacing `NN` and `<module>`.

## Phase 08 prompt

Convert Phase 08 (Teaching Modes & Prompt Engineering) tutorial notebook to production.

Read these four files first:

1. `production-ready.md` → Phase 08 section — confirms: add `beginner.j2`, `expert.j2`,
   `quiz.j2`, `explain.j2` (and also `_base.j2`, `interview.j2`, `lecture_notes.j2` from
   the tutorial); extend `src/mrta/prompts/__init__.py` with a `MODES` constant
2. `notebook-to-production-steps.md` → Phase 07 is the last completed phase
3. Tutorial notebook: `notebooks/tutorials/2026-05-25-phase08-teaching-modes-and-prompts.ipynb`
4. Production notebook: `notebooks/production/2026-05-25-phase08-teaching-modes-and-prompts.ipynb`

Also inspect `src/mrta/prompts/` — it currently contains `__init__.py` (with `load_prompt`
already implemented) and `rag.j2`. The six new template files do not exist yet.

**Four interface differences** — tutorial inline code does not match the production setup:

- Tutorial uses `.jinja2` extension (`beginner.jinja2`, `_base.jinja2`, etc.) →
  Production uses `.j2` (matching existing `rag.j2`). All template files must be `.j2`.
- Tutorial names the graduate-level template `graduate.jinja2` →
  Production target in `production-ready.md` is `expert.j2`. Use `expert.j2`. The content
  (behavior block) is the same as the tutorial's `graduate.jinja2`.
- Tutorial defines `render_prompt(mode, *, question, chunks)` with a `FileSystemLoader`
  and a `MODES` dict inline in the notebook →
  Production: `load_prompt(name, **kwargs)` already exists in `src/mrta/prompts/__init__.py`
  and uses `PackageLoader`. Do NOT reimplement it. The `MODES` dict moves to `__init__.py`
  as a public constant so callers (Streamlit app, tests) can enumerate available modes.
- Tutorial has no `explain.j2` →
  Production needs `explain.j2` for figure explanation via `VLMClient.caption(image, prompt)`.
  This template is standalone (does not extend `_base.j2`) because it is a short imperative
  instruction, not a RAG prompt. It takes optional `level` and `question` variables.

**`load_prompt` is already done** — do not rewrite `src/mrta/prompts/__init__.py` from scratch.
Only add `MODES` and update `__all__`.

5. Extend `src/mrta/prompts/__init__.py` — add `MODES` constant and export it:

```python
MODES: dict[str, str] = {
    "default":       "_base",
    "beginner":      "beginner",
    "expert":        "expert",
    "interview":     "interview",
    "quiz":          "quiz",
    "lecture_notes": "lecture_notes",
    "explain":       "explain",
}

__all__ = ["load_prompt", "MODES"]
```

`MODES` maps the user-facing mode name to the template base name (without `.j2`). The
Streamlit app can do `from mrta.prompts import MODES` instead of maintaining its own dict.

6a. Create `src/mrta/prompts/_base.j2` — the shared parent template. Copy the content
from tutorial cell [4] (the `_base.jinja2` string), converting `{% block role %}` /
`{% block behavior %}` / `{% block format %}` blocks intact. This template is the grounding
wrapper all RAG-mode templates extend.

Full content:

```jinja2
{% block role %}You are a research and teaching assistant.{% endblock %}
{% block behavior %}Answer the user's question grounded in the provided context.{% endblock %}

Use ONLY the context below. If the context is insufficient, say so.
Cite the source page for every claim, like [page 3] or [page 3, 5].

--- CONTEXT ---
{% for c in chunks %}
[chunk {{ loop.index }} | {{ c.source }} | page {{ c.page }}]
{{ c.text }}

{% endfor %}
--- QUESTION ---
{{ question }}

{% block format %}--- ANSWER (with [page X] citations) ---{% endblock %}
```

6b. Create `src/mrta/prompts/beginner.j2` — extends `_base.j2`:

```jinja2
{% extends "_base.j2" %}
{% block behavior %}Explain like the reader is new to this topic.
- Use plain language and concrete analogies.
- Define jargon the first time it appears.
- Keep paragraphs short.{% endblock %}
{% block format %}--- ANSWER (plain language, with [page X] citations) ---{% endblock %}
```

6c. Create `src/mrta/prompts/expert.j2` — extends `_base.j2`. Uses the same behavior
content as the tutorial's `graduate.jinja2` (graduate-level ML background):

```jinja2
{% extends "_base.j2" %}
{% block behavior %}Assume the reader has a graduate-level ML background.
- Use precise terminology (attention, softmax temperature, layer norm, etc.).
- Reference equations and dimensions when relevant.
- Compare to alternative approaches when the paper does.{% endblock %}
{% block format %}--- ANSWER (graduate level, with [page X] citations) ---{% endblock %}
```

6d. Create `src/mrta/prompts/interview.j2` — extends `_base.j2`:

```jinja2
{% extends "_base.j2" %}
{% block behavior %}Frame the answer as you would in an ML system-design interview.
- State the problem and motivation in one sentence.
- Identify two or three design choices and their tradeoffs.
- Discuss complexity (time, memory, compute) where relevant.
- End with one extension or follow-up question you would expect from the interviewer.{% endblock %}
{% block format %}--- ANSWER (interview style, with [page X] citations) ---{% endblock %}
```

6e. Create `src/mrta/prompts/quiz.j2` — extends `_base.j2`:

```jinja2
{% extends "_base.j2" %}
{% block behavior %}Generate exactly 5 multiple-choice quiz questions about the topic.
- Each has 4 options (A, B, C, D) and exactly one correct answer.
- Cover different aspects: motivation, mechanism, results, limitations, math.
- After listing all 5, output an "ANSWER KEY" section with the correct letter and a
  1-sentence justification citing pages.{% endblock %}
{% block format %}--- QUIZ ---{% endblock %}
```

6f. Create `src/mrta/prompts/lecture_notes.j2` — extends `_base.j2`:

```jinja2
{% extends "_base.j2" %}
{% block behavior %}Produce study-style lecture notes.
- Begin with "Key Terms" (5–8 bullets, each defined in one line).
- Then "Mechanism" (numbered steps).
- Then "Results" (one paragraph of numbers + their context).
- Then "Open questions / limitations" (3 bullets).{% endblock %}
{% block format %}--- LECTURE NOTES (with [page X] citations) ---{% endblock %}
```

6g. Create `src/mrta/prompts/explain.j2` — standalone (does NOT extend `_base.j2`).
Used by `VLMClient.caption(image, prompt=load_prompt("explain", ...))`. Variables:
`level` (optional, defaults to `"graduate student"`) and `question` (optional):

```jinja2
You are a research assistant explaining a scientific figure.
Describe what this figure shows concretely. Reference specific visual elements
(axes, curves, arrows, annotations) and connect them to the broader concept.
Explain at {{ level | default("graduate student") }} level.
{% if question %}Focus especially on: {{ question }}{% endif %}
If you cannot determine what the figure shows, say so explicitly.
```

7. Export `MODES` from `src/mrta/__init__.py` — add to the imports and `__all__`:

```python
from mrta.prompts import load_prompt, MODES
```

8. Update production notebook — replace inline blocks with imports:

   - Cell [1]: update header from "Production imports" → "Production imports (active)";
     replace the stub code block with:

     ```python
     from mrta.prompts import load_prompt, MODES
     ```

   - Cell [6]: replace the inline `(PROMPTS / "_base.jinja2").write_text(...)` call with:

     ```python
     # Template: see src/mrta/prompts/_base.j2
     ```

   - Cell [8]: replace the inline block that writes all five `.jinja2` files with:

     ```python
     # Templates: see src/mrta/prompts/ (beginner, expert, interview, quiz, lecture_notes)
     ```

   - Cell [10]: replace the `render_prompt` definition (FileSystemLoader + MODES dict + function)
     with:

     ```python
     from mrta.prompts import load_prompt, MODES

     # Smoke test:
     preview = load_prompt("beginner", question="What is attention?", chunks=[
         {"source": "aiayn.pdf", "page": 4, "text": "Scaled dot-product attention computes ..."}
     ])
     print(preview[:600])
     ```

   - Cell [12] (inline VectorStore + SentenceTransformer + `ask()` demo): keep inline.
     This cell requires a live Ollama server and is the teaching content for 8.5.

   - All remaining cells: keep inline — prompt-quality checks (8.6), heuristics (8.7),
     "What you learned", and exercises are all teaching content, not library code.

9. Write tests in `tests/unit/test_prompts.py`:

   - `load_prompt("rag", question="Q", chunks=[])` returns non-empty string containing "Q"
   - `load_prompt("beginner", question="test question", chunks=[])` returns non-empty string
     containing "test question"
   - `load_prompt("expert", question="Q", chunks=[])` returns non-empty string
   - `load_prompt("quiz", question="Q", chunks=[])` returns non-empty string containing "QUIZ"
   - `load_prompt("explain")` returns non-empty string (no required kwargs)
   - `load_prompt("explain", level="high school student")` includes "high school student"
   - `load_prompt("nonexistent")` raises `jinja2.exceptions.TemplateNotFound`
   - `MODES` is a dict with at least these keys: `"beginner"`, `"expert"`, `"quiz"`,
     `"lecture_notes"`, `"interview"`, `"explain"`
   - Each value in `MODES` corresponds to a template file that can be loaded:
     `load_prompt(name, question="Q", chunks=[])` succeeds for all RAG modes;
     `load_prompt("explain")` succeeds for the explain mode

10. Run `MRTA_ENV=test pytest -q` — all tests must pass.

Update documents:

11. `production-ready.md`:
    - Phase 08 table: mark all rows ✅ done; add rows for `_base.j2`, `interview.j2`,
      `lecture_notes.j2` (tutorial extras, included beyond minimum spec)
    - Library map: update `prompts/` line to `✅ complete (all templates done)`

12. `notebook-to-production-steps.md` → add Phase 08 section with:
    - "What's extracted" table (tutorial cell → production file)
    - "What stays inline" table (cell [12] demo, 8.6, 8.7, exercises)
    - Running notebook cell status table
    - Interface differences table (columns `Tutorial | Production | Reason`) — one row per
      mismatch: `.jinja2` → `.j2`, `graduate` → `expert`, `FileSystemLoader` → `PackageLoader`,
      `MODES` dict inline → `MODES` constant in `__init__.py`, `explain.j2` added (no tutorial
      equivalent)
    - Concept note: why `_base.j2` uses Jinja2 block inheritance rather than a Python string
      concatenation or f-string approach (block inheritance lets each mode override only the
      behavioral section while the grounding rule, context block, and citation instruction stay
      identical across all modes — a single change to `_base.j2` propagates to all modes without
      touching the mode files; f-strings would require every mode to duplicate the full template)

Wrap up: update memory, suggest git commit commands.

After committing, update `CHANGELOG.md`: read the file, run `git log --oneline -10`
for the Phase 08 hash, read the Phase 08 section of `notebook-to-production-steps.md`
and `tests/unit/test_prompts.py`. Insert a new entry at the top (above `## [Phase 07]`):
`## [Phase 08] — Teaching Modes & Prompt Engineering — YYYY-MM-DD` with commit hash,
notebook paths, changed files table (columns `File | Change | Notes`), and tests table
(columns `Test class | Test | Assertion`).
Do not modify existing entries.
Suggest: `git add CHANGELOG.md && git commit -m "docs: update CHANGELOG for Phase 08"`

---

## Phase 07 prompt

Convert Phase 07 (Figure Extraction & VLM) tutorial notebook to production.

Read these four files first:

1. `production-ready.md` → Phase 07 section — confirms: extract `FigureRecord` schema,
   `extract_figures`, `CLIPEmbedder`, `VLMClient`; `FigureRecord` spec and `CLIPEmbedder`
   / `VLMClient` interfaces are defined there
2. `notebook-to-production-steps.md` → Phase 06 is the last completed phase
3. Tutorial notebook: `notebooks/tutorials/2026-05-25-phase07-figure-extraction-and-vlm.ipynb`
4. Production notebook: `notebooks/production/2026-05-25-phase07-figure-extraction-and-vlm.ipynb`

Also inspect `src/mrta/multimodal/` (has `__init__.py` only — `clip_embedder.py` and
`vlm_client.py` do not exist yet) and `src/mrta/ingestion/` (no `figure_extractor.py` yet).

**Five interface differences** — tutorial inline code does not match the target interfaces
in `production-ready.md`; apply these when implementing:

- Tutorial `Figure` schema has `figure_id`, `image_path`, `bbox`, `caption` →
  Production `FigureRecord` has `doc_id`, `source`, `page`, `figure_index: int`,
  `image_bytes: bytes`. No `figure_id` field (derive as `f"{doc_id}_p{page}_fig{figure_index}"`
  when needed). No disk path — store bytes directly.
- Tutorial `extract_figures` writes PNG files to `FIG_DIR` (disk) and stores the path →
  Production: skip disk writes; store `pix.tobytes("png")` in `FigureRecord.image_bytes`.
- Tutorial `doc_id_of(path)` re-implements SHA-1 hashing inline →
  Production: import `_doc_id` from `mrta.ingestion.pdf_loader` (same logic; no duplication).
- Tutorial `clip_image_embedding(path: str)` takes a file path →
  Production `CLIPEmbedder.embed_image(image: Image.Image)` takes a PIL `Image`; the
  caller converts via `figure.to_pil()`.
- Tutorial `vlm_explain_figure_ollama(image_path: str, ...)` takes a file path, hardcodes
  `"llava:7b"` → Production `VLMClient.caption(image: Image.Image, prompt=None)` takes a
  PIL `Image`, reads `settings.ollama_vlm_model` (currently `"qwen2.5vl:7b"`).

**Out of scope for Phase 07:** The tutorial's cell [14] sketches a `POST /explain_figure`
FastAPI route. `production-ready.md` does NOT list that route in Phase 07's extraction plan.
Do not add it now — that wiring requires a figure store on `app.state` that hasn't been
designed yet.

5. Add `FigureRecord` to `src/mrta/core/schemas.py`:

```python
class FigureRecord(BaseModel):
    doc_id: str
    source: str
    page: int
    figure_index: int        # 1-indexed per page
    image_bytes: bytes

    def to_pil(self) -> "Image.Image":
        import io
        from PIL import Image
        return Image.open(io.BytesIO(self.image_bytes))
```

The `to_pil()` method is needed because `CLIPEmbedder.embed_image` and `VLMClient.caption`
both take `PIL.Image.Image`, not bytes.

6a. Create `src/mrta/ingestion/figure_extractor.py` — `extract_figures`:

```python
from pathlib import Path
import fitz  # PyMuPDF

from mrta.core.schemas import FigureRecord
from mrta.ingestion.pdf_loader import _doc_id

def extract_figures(pdf_path: Path) -> list[FigureRecord]:
    doc = fitz.open(pdf_path)
    did = _doc_id(pdf_path)
    figs: list[FigureRecord] = []
    for page_num, page in enumerate(doc, start=1):
        for idx, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            figs.append(FigureRecord(
                doc_id=did,
                source=Path(pdf_path).name,
                page=page_num,
                figure_index=idx,
                image_bytes=pix.tobytes("png"),
            ))
            pix = None
    return figs
```

6b. Create `src/mrta/multimodal/clip_embedder.py` — `CLIPEmbedder`:

```python
class CLIPEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        # lazy-import open_clip (optional dep in [multimodal] group)
        import open_clip, torch
        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model = model.eval()
        self._preprocess = preprocess

    def embed_image(self, image: "Image.Image") -> np.ndarray:
        """Embed a PIL image. Returns float32 L2-normalised (512,) vector."""

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text string. Returns float32 L2-normalised (512,) vector."""
```

Both methods use `@torch.no_grad()` internally. Return shape `(512,)` for ViT-B-32.
L2-normalise before returning: `feat = feat / feat.norm(dim=-1, keepdim=True)`.

6c. Create `src/mrta/multimodal/vlm_client.py` — `VLMClient`:

```python
class VLMClient:
    def __init__(self, model: str | None = None) -> None:
        # reads settings.ollama_vlm_model if model is None
    def caption(self, image: "Image.Image", prompt: str | None = None) -> str:
        # convert image → PNG bytes → base64
        # call ollama.chat with images=[img_b64]
        # default prompt: "Explain this figure for a graduate student. Be concrete."
```

Reads `settings.ollama_vlm_model` by default (currently `"qwen2.5vl:7b"`). Converts
PIL image to PNG bytes with `io.BytesIO`, then base64-encodes. Uses same `ollama.chat`
pattern as `LLMClient` (already in `src/mrta/core/llm.py`).

7. Update exports:
   - `src/mrta/multimodal/__init__.py` — export `CLIPEmbedder`, `VLMClient`; define `__all__`
   - `src/mrta/ingestion/__init__.py` — add `extract_figures` alongside `load_pdf`, `chunk_pdf`
   - `src/mrta/__init__.py` — add `FigureRecord`, `extract_figures`, `CLIPEmbedder`,
     `VLMClient` to imports and `__all__`

8. Update production notebook — replace inline blocks with imports:
   - Cell [1]: update header "Production imports" → "Production imports (active)"
   - Cell [5]: replace the inline `Figure` class, `doc_id_of`, and `extract_figures`
     definitions with the following (keep the print statement):

     ```python
     from mrta.core.schemas import FigureRecord
     from mrta.ingestion.figure_extractor import extract_figures
     figures = extract_figures(Path("data/sample/attention_is_all_you_need.pdf"))
     print(f"Extracted {len(figures)} figures")
     ```

   - Cell [9]: replace raw `open_clip` setup + inline `clip_image_embedding` /
     `clip_text_embedding` functions with the following (keep the `fig_embs` line):

     ```python
     from mrta.multimodal.clip_embedder import CLIPEmbedder
     clip = CLIPEmbedder()
     fig_embs = np.stack([clip.embed_image(f.to_pil()) for f in figures]) if figures else np.array([])
     ```

   - Cell [13]: replace `vlm_explain_figure_ollama` definition with the following;
     keep the demo call inline — it needs a live Ollama server:

     ```python
     from mrta.multimodal.vlm_client import VLMClient
     vlm = VLMClient()  # reads settings.ollama_vlm_model
     ```

   - Cells [7], [11], [15], [17]: keep inline — display demos, retrieval loop, HF
     fallback (commented out), and honest-limitations notes.

9. Write tests in `tests/unit/test_figure_extractor.py`:
   - Load `tests/fixtures/sample.pdf` and call `extract_figures`
   - Result is a `list` of `FigureRecord` instances (0 or more — sample.pdf may have
     no embedded raster images; test for type, not count > 0)
   - Any returned record has non-empty `image_bytes`
   - Any returned record has `page >= 1` and `figure_index >= 1`
   - `to_pil()` returns a `PIL.Image.Image`
   - Use `pytest.importorskip("fitz")` to skip if PyMuPDF not installed

   Write tests in `tests/unit/test_clip_embedder.py`:
   - Use `pytest.importorskip("open_clip")` to skip if not installed
   - Create a 1×1 white PIL image; `clip.embed_image(img)` returns shape `(512,)`
   - Embedding is float32
   - L2 norm is `~1.0` (within 1e-5)
   - `clip.embed_text("attention mechanism")` returns shape `(512,)` and norm `~1.0`
   - Image and text embeddings of matching content have positive dot product

10. Run `MRTA_ENV=test pytest -q` — all tests must pass.

Update documents:

11. `production-ready.md`:
    - Library map: update `ingestion/figure_extractor.py` from `stub` → `✅ done`;
      update `multimodal/clip_embedder.py` and `multimodal/vlm_client.py` from `stub` → `✅ done`;
      update `core/schemas.py` note to include `FigureRecord`
    - Phase 07 table: mark all three rows ✅ done

12. `notebook-to-production-steps.md` → add Phase 07 section with:
    - "What's extracted" table (tutorial cell → production file)
    - "What stays inline" table (cells [7], [11], [13] demo call, [15], [17])
    - Running notebook cell status table
    - Interface differences table (columns `Tutorial | Production | Reason`) — one row
      per mismatch
    - Concept note: why `CLIPEmbedder` and text `Embedder` are separate classes
      (different model families, different normalisation spaces — CLIP's shared
      image-text space is not compatible with sentence-transformer text embeddings;
      mixing them in a single class would require two model instances and two index
      namespaces, which is more complex than two small wrapper classes)

Wrap up: update memory, suggest git commit commands.

After committing, update `CHANGELOG.md`: read the file, run `git log --oneline -10`
for the Phase 07 hash, read the Phase 07 section of `notebook-to-production-steps.md`
and both test files. Insert a new entry at the top (above `## [Phase 06]`):
`## [Phase 07] — Figure Extraction & VLM — YYYY-MM-DD` with commit hash, notebook
paths, changed files table (columns `File | Change | Notes`), and tests table
(columns `Test class | Test | Assertion`).
Do not modify existing entries.
Suggest: `git add CHANGELOG.md && git commit -m "docs: update CHANGELOG for Phase 07"`

---

## Phase 06 prompt

Convert Phase 06 (Streamlit Frontend) tutorial notebook to production.

Read these four files first:

1. `production-ready.md` → Phase 06 section — confirms: **nothing to extract** to
   `src/mrta/`; all UI code belongs in `apps/streamlit/app.py`; the one rule is that
   `app.py` must call the REST API via `httpx` and must NOT import from `apps.api`
2. `notebook-to-production-steps.md` → Phase 05 is the last completed phase
3. Tutorial notebook: `notebooks/tutorials/2026-05-25-phase06-streamlit-frontend.ipynb`
4. Production notebook: `notebooks/production/2026-05-25-phase06-streamlit-frontend.ipynb`

Also read `apps/streamlit/app.py` — it is currently a 7-line scaffold stub.

**Three interface mismatches** — the tutorial's source code uses the old API contract;
fix them when writing `app.py`:

- `payload = {"question": ..., "k": k}` → `{"question": ..., "top_k": k}` (matches
  `AskRequest.top_k`; the field is named `top_k`, not `k`)
- `resp["cited_pages"]` → does not exist in `AskResponse`; derive cited pages from
  `resp["sources"]` instead: `sorted({s["page"] for s in resp["sources"]})`
- `resp["model"]` → does not exist in `AskResponse`; remove that caption line
- `resp["retrieved"]` → use `resp["sources"]`; each item has `page`, `chunk_id`,
  `preview` (no `score` field — remove `score` from the chunks expander)

5. No library schemas or functions to extract in this phase. `apps/streamlit/app.py`
   calls the REST API only — zero `from mrta.*` imports needed.

6. Replace `apps/streamlit/app.py` with the full implementation. Start from the
   tutorial's `streamlit_src` string (cell [3]), apply the four fixes above, and write
   it as a clean Python file (not a string-in-a-variable). Structure:

   ```python
   # --- page config ----------------------------------------------------------
   # --- sidebar: upload + doc list -------------------------------------------
   # --- main panel: mode + question + ask ------------------------------------
   # --- response: answer + cited pages + retrieved chunks expander -----------
   ```

   Keep the teaching-modes radio exactly as in the tutorial (six modes, `mode_prefix`
   dict). Keep the graceful fallback for when the backend is unreachable. Keep the
   `disabled=not question` guard on the Ask button.

7. Update production notebook — cell [5] contains the same `streamlit_src` string
   variable. Replace it with a single comment:
   `# Full implementation: see apps/streamlit/app.py`
   Update cell [1] header from "Production note" to "Production note (active)".

8. **No new tests.** Streamlit UI has no unit-testable logic — the phase contains no
   functions to unit-test; all correctness is covered by the Phase 05 API tests.
   Note this in `notebook-to-production-steps.md`.

Update documents:

9. `production-ready.md` → Phase 06 section: mark the `apps/streamlit/app.py` item
   `✅ done`.

10. `notebook-to-production-steps.md` → add Phase 06 section with:
    - "What's extracted" table (tutorial cell → production file in `apps/streamlit/`)
    - "What stays inline" (nothing — all cells are markdown or a demo string variable)
    - Running notebook cell status table
    - Interface fixes table (columns `Tutorial field | Production field | Reason`) —
      one row per mismatch fixed
    - Concept note: why the Streamlit app calls the REST API rather than importing
      `mrta.*` directly (frontend/backend separation — independently deployable;
      `httpx` is the anti-corruption layer; the API version boundary means the UI
      doesn't break when the library refactors internally)

Wrap up: update memory, suggest git commit commands.

After committing, update `CHANGELOG.md`: read the file, run `git log --oneline -10`
for the Phase 06 hash, read the Phase 06 section of `notebook-to-production-steps.md`.
Insert a new entry at the top (above `## [Phase 05]`):
`## [Phase 06] — Streamlit Frontend — YYYY-MM-DD` with commit hash, notebook paths,
changed files table (columns `File | Change | Notes`), and a note in place of the
tests table explaining why no new tests were added (Streamlit UI has no unit-testable
functions; correctness covered by Phase 05 API tests).
Do not modify existing entries.
Suggest: `git add CHANGELOG.md && git commit -m "docs: update CHANGELOG for Phase 06"`

---

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
