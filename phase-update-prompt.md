# Phase Update Prompt

Paste this at the start of any notebook-to-production session, replacing `NN` and `<module>`.

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

```
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
