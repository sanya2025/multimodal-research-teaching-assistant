"""Integration tests for production ingestion of every retrieval artifact.

Covers the gap PR6 closed: an uploaded PDF must build the text, caption and
CLIP indices, not just the text index, so a newly ingested document is
immediately usable by the canonical Text + Caption + CLIP query path.

Real PDFs are synthesised with PyMuPDF; embedding and captioning are faked, so
nothing here downloads a model. No benchmark or evaluation path is referenced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mrta.core.schemas import EvidenceRecord, FigureRecord, VisualRecord
from mrta.ingestion.document_indexer import (
    CAPTION_INDEX_NAME,
    CAPTION_SOURCE_FALLBACK,
    CAPTION_SOURCE_VLM,
    CLIP_INDEX_NAME,
    TEXT_INDEX_NAME,
    build_caption_evidence,
    build_visual_record,
    index_document,
)
from mrta.retrieval.canonical_pipeline import (
    evidence_record_to_candidate,
    retrieve_multimodal,
    visual_record_to_candidate,
)
from mrta.retrieval.fusion import canonical_identity

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

# faiss is imported inside the fixture, never at module scope. Importing it at
# collection time claims the OpenMP runtime before any test runs, which makes a
# later real open_clip model build segfault on macOS (see CLIPEmbedder.warmup).
# tests/unit/test_caption_store.py and test_image_store.py guard it the same way.


# ---------------------------------------------------------------------------
# Fakes — deterministic, no model downloads
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic hash-based embedder standing in for nomic/MiniLM."""

    dim = 16
    model_name = "fake-embedder"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in (text or "").lower().split():
                out[i][hash(token) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class FakeCLIP:
    """Deterministic CLIP stand-in — no open_clip weights, no torch, no network.

    ``warmup`` is deliberately a no-op. ImageStore calls it to force torch's
    OpenMP runtime up before FAISS claims it (see CLIPEmbedder.warmup), but that
    ordering only matters when torch is actually in play. Importing torch here
    instead *creates* the hazard: it puts torch before FAISS in a process that
    later builds a real open_clip model, which is the exact sequence that
    segfaults on macOS. Keeping this fake torch-free leaves the invariant to the
    tests that use the real embedder.
    """

    dim = 8
    model_name = "fake-clip"

    def warmup(self) -> None:
        return None

    def _vec(self, seed: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for token in seed.lower().split():
            vec[hash(token) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else np.ones(self.dim, dtype="float32") / np.sqrt(self.dim)

    def embed_image(self, image) -> np.ndarray:
        return self._vec(str(image))

    def embed_text(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeAnalyzer:
    """VisualAnalyzer stand-in that populates caption fields."""

    def __init__(self, caption: str | None = "A chart showing results.", raises=False) -> None:
        self.caption = caption
        self.raises = raises
        self.calls = 0

    def analyze_evidence(self, record: EvidenceRecord):
        self.calls += 1
        if self.raises:
            raise RuntimeError("VLM unreachable")
        if self.caption is not None:
            record.caption = self.caption
            record.detailed_description = f"{self.caption} Detailed view."
            record.visual_type = "bar_chart"
        return None


def _make_pdf(path: Path, *, with_image: bool = True, pages: int = 2) -> Path:
    """Write a small real PDF, optionally containing an embedded raster image."""
    doc = fitz.open()
    for n in range(pages):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {n + 1}: attention mechanisms and softmax scaling.")
        if with_image and n == 0:
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
            pix.set_rect(pix.irect, (200, 60, 60))
            page.insert_image(fitz.Rect(200, 200, 300, 300), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def stores():
    pytest.importorskip("faiss", reason="faiss-cpu not installed")
    from mrta.retrieval.caption_store import CaptionVectorStore
    from mrta.retrieval.image_store import ImageStore
    from mrta.retrieval.vector_store import VectorStore

    embedder = FakeEmbedder()
    clip = FakeCLIP()
    return {
        "text": VectorStore(embedder),
        "caption": CaptionVectorStore(embedder),
        "image": ImageStore(clip),
        "embedder": embedder,
        "clip": clip,
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolated working directory so assets land under ./data, as in production."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Artifacts produced per uploaded PDF
# ---------------------------------------------------------------------------


class TestArtifactCreation:
    def test_text_index_created(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf", with_image=False)
        result = index_document(pdf, text_store=stores["text"], store_root=workspace / "vs")
        assert result.n_chunks > 0
        assert (workspace / "vs" / TEXT_INDEX_NAME / "index.faiss").exists()

    def test_figure_pdf_creates_caption_record(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf")
        analyzer = FakeAnalyzer()
        result = index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            analyzer=analyzer,
            store_root=workspace / "vs",
        )
        assert result.n_figures >= 1
        assert result.n_caption_records >= 1
        assert stores["caption"].size >= 1
        assert analyzer.calls == result.n_figures

    def test_figure_pdf_creates_clip_visual_record(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf")
        result = index_document(
            pdf,
            text_store=stores["text"],
            image_store=stores["image"],
            store_root=workspace / "vs",
        )
        assert result.n_visual_records >= 1
        assert stores["image"].size >= 1

    def test_figure_asset_written_under_data_root(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf")
        index_document(
            pdf,
            text_store=stores["text"],
            image_store=stores["image"],
            store_root=workspace / "vs",
        )
        assets = list((workspace / "data" / "figures").glob("*.png"))
        assert assets, "figure PNG should be written under data/figures"

    def test_all_three_indices_built_in_one_pass(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf")
        result = index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            analyzer=FakeAnalyzer(),
            store_root=workspace / "vs",
        )
        assert result.n_chunks > 0
        assert result.n_caption_records >= 1
        assert result.n_visual_records >= 1
        assert result.visual_retrieval_available is True


# ---------------------------------------------------------------------------
# Canonical identity compatibility
# ---------------------------------------------------------------------------


class TestCanonicalIdentityFromIngestion:
    def test_caption_and_clip_records_share_canonical_identity(self) -> None:
        """The two records built from one FigureRecord must merge under fusion."""
        figure = FigureRecord(
            doc_id="docabc",
            source="doc.pdf",
            page=3,
            figure_index=2,
            image_bytes=b"\x89PNG\r\n\x1a\n",
            nearby_text="Figure 2: results.",
        )
        evidence, _ = build_caption_evidence(figure, "data/figures/x.png", analyzer=None)
        visual = build_visual_record(figure, "data/figures/x.png")

        caption_identity = canonical_identity(evidence_record_to_candidate(evidence, 0.9, 1))
        clip_identity = canonical_identity(visual_record_to_candidate(visual, 0.3, 1))
        assert caption_identity == clip_identity
        assert caption_identity[0] == "figure"

    def test_derived_figure_id_matches_retrieval_derivation(self) -> None:
        figure = FigureRecord(
            doc_id="docabc",
            source="doc.pdf",
            page=7,
            figure_index=4,
            image_bytes=b"\x89PNG\r\n\x1a\n",
        )
        assert build_visual_record(figure, "data/figures/x.png").figure_id == "p7_f4"

    def test_ingested_document_merges_streams_end_to_end(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf")
        index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            analyzer=FakeAnalyzer(caption="attention softmax chart"),
            store_root=workspace / "vs",
        )
        final, diagnostics = retrieve_multimodal(
            "attention softmax",
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
        )
        figures = [c for c in final if c.candidate.evidence_type == "figure"]
        assert figures, "the ingested figure should be retrievable"
        assert set(figures[0].candidate.modality_sources) == {"caption", "clip"}
        assert diagnostics.degraded_streams == {}


# ---------------------------------------------------------------------------
# Immediate queryability — the actual gap that was closed
# ---------------------------------------------------------------------------


class TestImmediateQueryability:
    def test_newly_ingested_document_is_queryable_via_all_streams(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf")
        index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            analyzer=FakeAnalyzer(),
            store_root=workspace / "vs",
        )
        final, diagnostics = retrieve_multimodal(
            "attention mechanisms",
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
        )
        assert final
        assert diagnostics.stream_sizes["text"] > 0
        assert diagnostics.stream_sizes["caption"] > 0
        assert diagnostics.stream_sizes["clip"] > 0

    def test_no_benchmark_paths_required(self, workspace, stores) -> None:
        """Ingestion and query must not touch data/eval or any manifest."""
        pdf = _make_pdf(workspace / "doc.pdf")
        index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            analyzer=FakeAnalyzer(),
            store_root=workspace / "vs",
        )
        assert not (workspace / "data" / "eval").exists()
        final, _ = retrieve_multimodal(
            "attention",
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
        )
        assert final

    def test_production_code_does_not_import_benchmark_scripts(self) -> None:
        import mrta.generation.canonical_rag as gen
        import mrta.ingestion.document_indexer as indexer
        import mrta.retrieval.canonical_pipeline as pipeline

        for module in (indexer, pipeline, gen):
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "scripts.run_eval" not in source
            assert "scripts/build_" not in source
            assert "data/eval" not in source


# ---------------------------------------------------------------------------
# Persistence across restart
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_indices_reopen_after_restart(self, workspace, stores) -> None:
        from mrta.retrieval.caption_store import CaptionVectorStore
        from mrta.retrieval.image_store import ImageStore
        from mrta.retrieval.vector_store import VectorStore

        pdf = _make_pdf(workspace / "doc.pdf")
        store_root = workspace / "vs"
        result = index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            analyzer=FakeAnalyzer(),
            store_root=store_root,
        )

        # Simulate a restart: rebuild every store from disk only.
        reopened_text = VectorStore.load(store_root / TEXT_INDEX_NAME, FakeEmbedder())
        reopened_caption = CaptionVectorStore.load(store_root / CAPTION_INDEX_NAME, FakeEmbedder())
        reopened_clip = ImageStore.load(store_root / CLIP_INDEX_NAME, FakeCLIP())

        assert len(reopened_text._chunks) == result.n_chunks
        assert reopened_caption.size == result.n_caption_records
        assert reopened_clip.size == result.n_visual_records

    def test_reopened_indices_still_serve_queries(self, workspace, stores) -> None:
        from mrta.retrieval.caption_store import CaptionVectorStore
        from mrta.retrieval.image_store import ImageStore
        from mrta.retrieval.vector_store import VectorStore

        pdf = _make_pdf(workspace / "doc.pdf")
        store_root = workspace / "vs"
        index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            analyzer=FakeAnalyzer(caption="attention softmax chart"),
            store_root=store_root,
        )
        final, diagnostics = retrieve_multimodal(
            "attention softmax",
            text_store=VectorStore.load(store_root / TEXT_INDEX_NAME, FakeEmbedder()),
            caption_store=CaptionVectorStore.load(store_root / CAPTION_INDEX_NAME, FakeEmbedder()),
            image_store=ImageStore.load(store_root / CLIP_INDEX_NAME, FakeCLIP()),
        )
        assert final
        assert diagnostics.stream_sizes["caption"] > 0
        assert diagnostics.stream_sizes["clip"] > 0

    def test_caption_records_retain_provenance_after_reload(self, workspace, stores) -> None:
        from mrta.retrieval.caption_store import CaptionVectorStore

        pdf = _make_pdf(workspace / "doc.pdf")
        store_root = workspace / "vs"
        index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            analyzer=FakeAnalyzer(caption="A results chart."),
            store_root=store_root,
        )
        reopened = CaptionVectorStore.load(store_root / CAPTION_INDEX_NAME, FakeEmbedder())
        record = reopened._records[0]
        assert record.caption == "A results chart."
        assert record.image_path and record.image_path.endswith(".png")
        assert record.figure_index is not None
        assert record.page >= 1


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


class TestIngestionDegradation:
    def test_pdf_without_figures_still_ingests(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "textonly.pdf", with_image=False)
        result = index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            analyzer=FakeAnalyzer(),
            store_root=workspace / "vs",
        )
        assert result.n_chunks > 0
        assert result.n_figures == 0
        assert result.visual_retrieval_available is False
        assert result.figure_failures == []

    def test_text_only_document_remains_searchable(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "textonly.pdf", with_image=False)
        index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            store_root=workspace / "vs",
        )
        final, _ = retrieve_multimodal(
            "attention mechanisms",
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
        )
        assert final
        assert all(c.candidate.evidence_type == "text" for c in final)

    def test_caption_index_failure_does_not_destroy_text_index(self, workspace, stores) -> None:
        class ExplodingCaptionStore:
            size = 0

            def add(self, records):
                raise RuntimeError("caption index write failed")

        pdf = _make_pdf(workspace / "doc.pdf")
        result = index_document(
            pdf,
            text_store=stores["text"],
            caption_store=ExplodingCaptionStore(),
            analyzer=FakeAnalyzer(),
            store_root=workspace / "vs",
        )
        assert result.n_chunks > 0
        assert (workspace / "vs" / TEXT_INDEX_NAME / "index.faiss").exists()
        assert "caption_index" in result.degraded
        assert result.n_caption_records == 0

    def test_clip_index_failure_does_not_destroy_text_index(self, workspace, stores) -> None:
        class ExplodingImageStore:
            size = 0

            def add_images(self, records):
                raise RuntimeError("clip embed failed")

        pdf = _make_pdf(workspace / "doc.pdf")
        result = index_document(
            pdf,
            text_store=stores["text"],
            image_store=ExplodingImageStore(),
            store_root=workspace / "vs",
        )
        assert result.n_chunks > 0
        assert (workspace / "vs" / TEXT_INDEX_NAME / "index.faiss").exists()
        assert "clip_index" in result.degraded

    def test_captioning_failure_falls_back_to_nearby_text(self, workspace, stores) -> None:
        """A VLM failure must not fabricate a caption or lose the figure."""
        figure = FigureRecord(
            doc_id="docabc",
            source="doc.pdf",
            page=1,
            figure_index=1,
            image_bytes=b"\x89PNG\r\n\x1a\n",
            nearby_text="Figure 1: throughput comparison.",
        )
        evidence, caption_source = build_caption_evidence(
            figure, "data/figures/x.png", analyzer=None
        )
        assert evidence.caption is None
        assert caption_source == CAPTION_SOURCE_FALLBACK
        assert evidence.retrieval_text() == "Figure 1: throughput comparison."

    def test_vlm_caption_recorded_as_vlm_provenance(self) -> None:
        figure = FigureRecord(
            doc_id="docabc",
            source="doc.pdf",
            page=1,
            figure_index=1,
            image_bytes=b"\x89PNG\r\n\x1a\n",
            nearby_text="nearby",
        )
        _, caption_source = build_caption_evidence(
            figure, "data/figures/x.png", analyzer=FakeAnalyzer(caption="A chart.")
        )
        assert caption_source == CAPTION_SOURCE_VLM

    def test_one_bad_figure_does_not_abort_the_document(self, workspace, stores) -> None:
        class PartiallyExplodingAnalyzer:
            def __init__(self):
                self.calls = 0

            def analyze_evidence(self, record):
                self.calls += 1
                raise RuntimeError("vlm exploded")

        pdf = _make_pdf(workspace / "doc.pdf")
        result = index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            analyzer=PartiallyExplodingAnalyzer(),
            store_root=workspace / "vs",
        )
        # Text survives and the failure is reported rather than swallowed.
        assert result.n_chunks > 0
        assert result.figure_failures
        assert "vlm exploded" in result.figure_failures[0]

    def test_visual_stores_absent_means_text_only_success(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf")
        result = index_document(pdf, text_store=stores["text"], store_root=workspace / "vs")
        assert result.n_chunks > 0
        assert result.n_caption_records == 0
        assert result.n_visual_records == 0


# ---------------------------------------------------------------------------
# Provenance retention
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_visual_record_carries_full_provenance(self) -> None:
        figure = FigureRecord(
            doc_id="docabc",
            source="doc.pdf",
            page=5,
            figure_index=3,
            image_bytes=b"\x89PNG\r\n\x1a\n",
        )
        visual: VisualRecord = build_visual_record(figure, "data/figures/a.png")
        assert visual.document_id == "docabc"
        assert visual.page == 5
        assert visual.figure_index == 3
        assert visual.figure_id == "p5_f3"
        assert visual.image_path == "data/figures/a.png"

    def test_caption_evidence_carries_full_provenance(self) -> None:
        figure = FigureRecord(
            doc_id="docabc",
            source="doc.pdf",
            page=5,
            figure_index=3,
            image_bytes=b"\x89PNG\r\n\x1a\n",
            nearby_text="context",
        )
        evidence, _ = build_caption_evidence(
            figure, "data/figures/a.png", analyzer=FakeAnalyzer(caption="A chart.")
        )
        assert evidence.doc_id == "docabc"
        assert evidence.source == "doc.pdf"
        assert evidence.page == 5
        assert evidence.figure_index == 3
        assert evidence.image_path == "data/figures/a.png"
        assert evidence.caption == "A chart."
        assert evidence.nearby_text == "context"

    def test_caption_source_persisted_on_the_record(self) -> None:
        """Provenance must live on the record, not only in the ingestion tally."""
        figure = FigureRecord(
            doc_id="docabc",
            source="doc.pdf",
            page=1,
            figure_index=1,
            image_bytes=b"\x89PNG\r\n\x1a\n",
            nearby_text="context",
        )
        vlm_evidence, _ = build_caption_evidence(
            figure, "data/figures/a.png", analyzer=FakeAnalyzer(caption="A chart.")
        )
        assert vlm_evidence.caption_source == CAPTION_SOURCE_VLM
        assert vlm_evidence.extraction_method == "raster_crop"

        fallback_evidence, _ = build_caption_evidence(figure, "data/figures/a.png", analyzer=None)
        assert fallback_evidence.caption_source == CAPTION_SOURCE_FALLBACK

    def test_extraction_method_recorded_on_visual_record(self) -> None:
        figure = FigureRecord(
            doc_id="docabc",
            source="doc.pdf",
            page=1,
            figure_index=1,
            image_bytes=b"\x89PNG\r\n\x1a\n",
        )
        assert build_visual_record(figure, "data/figures/a.png").extraction_method == "raster_crop"

    def test_provenance_survives_index_reload(self, workspace, stores) -> None:
        """caption_source and extraction_method must be recoverable after restart."""
        from mrta.retrieval.caption_store import CaptionVectorStore
        from mrta.retrieval.image_store import ImageStore

        pdf = _make_pdf(workspace / "doc.pdf")
        store_root = workspace / "vs"
        index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            image_store=stores["image"],
            analyzer=FakeAnalyzer(caption="A results chart."),
            store_root=store_root,
        )
        caption_record = CaptionVectorStore.load(
            store_root / CAPTION_INDEX_NAME, FakeEmbedder()
        )._records[0]
        assert caption_record.caption_source == CAPTION_SOURCE_VLM
        assert caption_record.extraction_method == "raster_crop"

        visual_record = ImageStore.load(store_root / CLIP_INDEX_NAME, FakeCLIP()).records[0]
        assert visual_record.extraction_method == "raster_crop"

    def test_caption_source_counts_reported(self, workspace, stores) -> None:
        pdf = _make_pdf(workspace / "doc.pdf")
        result = index_document(
            pdf,
            text_store=stores["text"],
            caption_store=stores["caption"],
            analyzer=FakeAnalyzer(),
            store_root=workspace / "vs",
        )
        assert sum(result.caption_sources.values()) == result.n_caption_records
        assert CAPTION_SOURCE_VLM in result.caption_sources
