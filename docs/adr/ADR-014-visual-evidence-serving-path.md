# ADR-014 — Repair the Visual Evidence Serving Path

**Date:** 2026-09-22
**Status:** Accepted
**Branch:** `fix/visual-evidence-serving-path`
**Relates to:**
[ADR-005](ADR-005-rag-architecture.md),
[ADR-008](ADR-008-multimodal-rag-architecture.md),
[ADR-009](ADR-009-canonical-retrieval-production-integration.md),
[ADR-013](ADR-013-modality-specific-retrieval-and-query-aware-reranking.md)

---

## Context

Asking *"Find the figure showing the Transformer model architecture and explain
what it depicts"* in **Multimodal RAG → Visual evidence** returned an answer with
no figure, and no Visual evidence panel at all — even though the figure was
correctly extracted, captioned and indexed, and its PNG was on disk at
`data/figures/attention_is_all_you_need_a639448e61_p3_f1.png`.

This ADR records four defects found behind that symptom. None of them changes the
architecture ADR-008, ADR-009 and ADR-013 describe. Each is a place where the
implementation did not deliver what those ADRs already decided, and in three
cases an exception handler turned a hard failure into silence.

### Defect 1 — the wrong CLIP encoder was wired into `ImageStore`

Two `CLIPEmbedder` classes exist, and their difference is deliberate and
documented: `mrta.retrieval.clip_embedder` loads CLIP through HuggingFace with
the correct QuickGELU activation, while `mrta.multimodal.clip_embedder` loads it
through open_clip with an activation mismatch. `ImageStore` is written against
the retrieval variant: it embeds figures **by path** and calls `warmup()` to
initialise torch's OpenMP runtime before FAISS.

The API lifespan constructed the `mrta.multimodal` variant and passed it to
`ImageStore`, which takes a PIL image and has no `warmup()`. Every
`add_images()` call therefore raised. Because ADR-009 §6 wraps each stream in
per-stream degradation, the exception was recorded as a degraded stream and
ingestion reported success — so `data/vector_store/clip_images/` was never
created, and ADR-009 §7 ("production ingestion builds all three indices") was
quietly untrue for the CLIP index.

### Defect 2 — the legacy retriever was handed an empty, never-loaded store

Teaching modes route to `MultimodalRetriever` rather than the canonical pipeline,
because their prompt templates consume `EvidenceRecord` lists (ADR-009 §3). That
routing is correct. But the lifespan built that retriever with a freshly
constructed `VisualVectorStore` and **no caption store**. `VisualVectorStore.load()`
is never called anywhere in the repository, and no ingestion path writes its
index, so the store was permanently empty. `search_with_scores()` returns `[]` on
an empty index without raising, so every teaching-mode query silently produced
zero visual evidence.

The two visual stores are not interchangeable: `VisualVectorStore` indexes
`EvidenceRecord`, while the persisted CLIP index holds `VisualRecord`, which is
deliberately narrower and carries no `source`. Pointing one at the other's index
fails schema validation.

### Defect 3 — persisted figures reached the VLM without their images

`MultimodalRAG` attached images only from `image_bytes`. Both the caption and
CLIP indices drop `image_bytes` at save time by design — they would add megabytes
per figure and are re-readable from disk. Every figure restored from a persisted
index therefore arrived with `image_bytes=None`, so an answer labelled
`retrieval_mode="multimodal"` was in fact produced without the model ever seeing
a figure, whenever the server had been restarted.

### Defect 4 — no endpoint could serve a figure

ADR-008 §5 kept binary image bytes out of `/ask` responses and stated that
"Streamlit fetches thumbnails via a separate `/figures` call". `/figures` was
only ever implemented to return VLM-written captions; it returns no pixels. No
endpoint served an image, and the Streamlit app contained no `st.image` call at
all. The `image_path` added by ADR-009 §5 is a server-side filesystem path a
browser cannot load. The design in ADR-008 §5 was therefore never completed.

## Decisions

### 1. One CLIP encoder per role, chosen by what consumes it

The API lifespan constructs `mrta.retrieval.clip_embedder.CLIPEmbedder` and uses
it for every visual index in the process. `mrta.multimodal.clip_embedder` remains
for `VisualVectorStore` and notebook use, but is not wired into the serving path.

This is the encoder whose identity ADR-013's v2 measurements were taken with, so
the served index and the evaluated index now use the same weights and the same
activation. That equivalence was previously assumed rather than enforced.

### 2. The legacy retriever reads the indices production ingestion writes

`MultimodalRetriever` receives the loaded caption store and a CLIP stream backed
by the persisted `clip_images` index, rather than an empty store of its own.

The type mismatch is bridged by `ImageStoreAdapter`
(`mrta.retrieval.image_store_adapter`), which presents an `ImageStore` through
the search surface the retriever expects and converts `VisualRecord` to
`EvidenceRecord`. It assigns `evidence_id` as `{doc_id}_p{page}_f{figure_index}`
— the same form the caption index uses — so RRF recognises a figure found by both
streams as one piece of evidence, preserving the canonical identity rule of
ADR-009 §1 and the deduplication rule of ADR-013 §3.

The adapter is read-only. Indexing stays with `ImageStore`, so exactly one
component writes the CLIP index.

`source` is resolved from the text index at query time, because `VisualRecord`
carries canonical identity but not the PDF filename. When it cannot be resolved
the citation shows `document_id`: degraded, but never an invented filename.

### 3. `MultimodalRetriever` depends on a protocol, not a concrete store

The visual stream is typed as `VisualSearchStore`, a `Protocol` satisfied by both
`VisualVectorStore` and `ImageStoreAdapter`. Structural typing keeps the
retriever from needing to know which implementation it was given, and makes the
next visual backend a new implementer rather than an edit to the retriever.

### 4. Figure images are loaded from `image_path` when bytes are absent

`MultimodalRAG` now takes each figure's image from `image_bytes` when present and
otherwise loads it from `image_path`, through the same `safe_image_path`
containment check the canonical path uses.

This does not weaken ADR-009 §5. A path is still never placed in the prompt; it
is resolved to actual pixels that are attached to the VLM call. The alternative —
leaving it as it was — meant `retrieval_mode="multimodal"` could not be trusted
to mean the model saw anything, which is a correctness claim, not a nicety.

### 5. Binary image bytes are served by a dedicated endpoint (completes ADR-008 §5)

`GET /figures/image?source=&page=&figure_index=` returns one figure PNG.

This implements, rather than revises, ADR-008 §5: `/ask` responses still carry no
base64 and no binary, and the bytes travel on a separate call. ADR-008 §5 named
`/figures` as that call; `/figures` is a VLM captioning endpoint whose cost is a
model invocation per figure, so the byte-serving role goes to its own route.

The path is derived server-side from the deterministic asset naming in
`figure_asset_path()` and is **never taken from the caller**. Accepting a
client-supplied path would make this a path-traversal surface; deriving it means
the only client input is a filename resolved against the index plus two positive
integers, after which `safe_image_path` applies the ADR-009 §5 containment check.

### 6. Visual citations carry their retrieved caption

`MultimodalCitation` on the legacy path now populates `image_path` and `caption`,
which ADR-009 §5 had added but only the canonical path filled. Clients display
the text the retriever actually scored the figure on, rather than commissioning a
fresh caption at display time. This keeps what the user reads consistent with
what the ranking saw, and removes a VLM call per figure per question.

### 7. Index repair is a script, not a migration

`scripts/rebuild_clip_serving_index.py` rebuilds `clip_images` from the caption
index, reusing the figure assets and canonical identities already recorded. It
needs CLIP weights but no Ollama.

Deployments whose CLIP index was never written under Defect 1 are repaired
without re-parsing PDFs or re-running VLM captioning. Once the wiring fix is
deployed, new uploads build the index correctly and the script is not part of
ingestion.

### 8. The lifespan warms torch's OpenMP runtime before touching FAISS

`apps/api/main.py` constructs the CLIP encoder and runs one forward pass
(`_warm_torch_runtime`) before it loads any FAISS index.

`faiss-cpu` and `torch` on macOS ARM each vendor their own `libomp.dylib`, and
whichever initialises the OpenMP runtime first wins. If FAISS gets there first,
torch's next forward pass dies with `SIGSEGV`. That is a signal, not an
exception: neither the lifespan's `except Exception` nor the per-stream
degradation of ADR-009 §6 can catch it. The worker simply disappears, and under
`uvicorn --reload` the parent respawns it, so a client sees only a reset
connection.

`CLIPEmbedder.warmup()` was written for precisely this and documents it, but it
was only ever reached from `ImageStore._ensure_index()` — which runs long after
the lifespan has already loaded the text index through FAISS. The protection
existed and was never in a position to work. Ordering the call correctly is the
whole fix.

Measured in this repository's Python 3.14 environment: 5/5 `SIGSEGV` with the
old ordering, 0/5 with the new one, and the full lifespan then starts cleanly
with both visual streams attached.

This is defence in depth, not the environment fix. Symlinking FAISS's `libomp`
to torch's copy, as the README describes, is what actually leaves one runtime in
the process; this ordering is what keeps a half-configured environment from
losing the API worker mid-upload. `OMP_NUM_THREADS=1` also avoids the crash but
serialises all OpenMP work for embedding and reranking, so it is documented as a
fallback rather than adopted.

Defect 1 is why this surfaced now. While `ImageStore` was raising on the wrong
encoder, CLIP never executed in the API process, so torch and FAISS never
contended. Making the CLIP index build for the first time is what exposed a
latent environment hazard — the bug was not introduced by that change, only
reached by it.

## Consequences

**Teaching modes gain visual evidence.** They were text-only in practice since
the multimodal path shipped. Any prior qualitative judgement of teaching-mode
output was made without figures and should be revisited.

**Reported multimodal answers become truthful.** Answers marked
`retrieval_mode="multimodal"` now attach images after a restart. Output for
figure-dependent questions will change — this is a correction, not a regression,
but it does move the baseline for any teaching-mode comparison.

**Frozen v2 results are unaffected.** The evaluation pipeline builds its own
indices under `data/eval/indices/` through `scripts/build_clip_image_index.py`,
which already used the correct encoder. No ADR-013 measurement changes, and the
ADR-012 regression gate is untouched.

**API startup is materially slower, and the Docker healthcheck was retuned.**
A working CLIP index means `ImageStore.load()` loads the CLIP encoder during
startup, which it never did while the index was failing to build. Measured cold
start in the API image went from roughly 35s to 69s, past the 30s
`start_period` in `compose.yaml` — so the first `docker compose up` after this
change reported the API as unhealthy and Streamlit refused to start behind its
`depends_on: service_healthy`. `start_period` is now 180s.

This is a real cost, not an accounting trick: the multimodal stack eagerly loads
two models (CLIP and the cross-encoder) before serving. Eager loading is the
existing decision from ADR-009 §8, which also records why it is disabled under
`MRTA_ENV=test`. Making these lazy would trade startup latency for first-query
latency and is a separate decision, not taken here.

**The serving CLIP index is new state.** `data/vector_store/clip_images/` now
exists and is loaded at startup. Its absence remains non-fatal per ADR-009 §6.

**A crash in the serving path is now survivable but not impossible.** The
ordering in §8 removes the failure this bug surfaced, and a startup-ordering
regression test guards it. An environment with two OpenMP runtimes is still
misconfigured, and the README check should be run against every venv the API is
launched from — not only the one used for development.

**Three swallowed exceptions are now understood but not removed.** Per-stream
degradation is still the right default — a missing caption index should not fail
a query. The cost is that a wiring error is indistinguishable from an absent
optional component. Surfacing degradation reasons in `/health` or an ingestion
summary is the obvious follow-up and is deliberately out of scope here. It is
recorded as [ADR-015](ADR-015-degradation-observability.md) (Proposed).

## Alternatives considered

**Route teaching modes through the canonical pipeline.** Rejected: their
templates consume `EvidenceRecord` lists, so routing them there would silently
change every rendered teaching prompt. That is a pedagogy change wearing a
plumbing change's clothes, and it belongs in its own ADR with its own evaluation.

**Make `VisualVectorStore.load()` read the `clip_images` directory.** Rejected:
the two record types differ for a reason, and teaching `VisualVectorStore` to
parse `VisualRecord` would give the system two writers for one index.

**Have `/upload` populate a second, `VisualVectorStore`-shaped CLIP index.**
Rejected: two CLIP indices over identical images, doubling ingestion cost and
creating a drift surface, to avoid one adapter.

**Return base64 image bytes in `/ask`.** Rejected: contradicts ADR-008 §5
directly, and makes every response pay for figures the caller may not render.

**Let Streamlit read `data/figures/` from disk.** Rejected: it only works when UI
and API share a filesystem, which the Docker compose setup does not guarantee,
and it puts asset-path trust decisions in the client.

## Related ADRs

- [ADR-005 — RAG Architecture](ADR-005-rag-architecture.md)
- [ADR-008 — Multimodal RAG Architecture](ADR-008-multimodal-rag-architecture.md)
- [ADR-009 — Canonical Retrieval in the Production Query Path](ADR-009-canonical-retrieval-production-integration.md)
- [ADR-013 — Modality-Specific Retrieval](ADR-013-modality-specific-retrieval-and-query-aware-reranking.md)
- [ADR-015 — Make Per-Stream Degradation Observable](ADR-015-degradation-observability.md): the follow-up this ADR scopes out
