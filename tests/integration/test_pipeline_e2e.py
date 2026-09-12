"""End-to-end integration tests for the PR6 production multimodal pipeline.

Covers the wiring from retrieval through fusion, reranking, generation and
citation assembly, plus the backward-compatibility contracts PR6 must preserve.

Every test here is fully mocked: no Ollama, no external API, no GPU, no model
download. The cross-encoder is represented by a fake whose ``rerank`` returns a
deterministic order, so ranking assertions are exact rather than approximate.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from apps.api.deps import get_canonical_stack, get_llm, get_retriever, get_store, get_vlm
from apps.api.main import app

from mrta.core.exceptions import LLMError
from mrta.core.schemas import Chunk, EvidenceRecord, VisualRecord
from mrta.generation.canonical_rag import (
    CanonicalMultimodalRAG,
    build_evidence_views,
    safe_image_path,
    verify_answer_citations,
)
from mrta.retrieval.canonical_pipeline import retrieve_multimodal

DOC = "attention_is_all_you_need"
PDF = "attention.pdf"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _chunk(page: int, idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{DOC}_p{page}_c{idx}",
        doc_id=DOC,
        source=PDF,
        page=page,
        text=text,
    )


def _caption_record(
    page: int, figure_index: int, caption: str | None = None, **kw
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"{DOC}_p{page}_f{figure_index}",
        doc_id=DOC,
        source=PDF,
        page=page,
        modality="image",
        figure_index=figure_index,
        caption=caption,
        **kw,
    )


class FakeTextStore:
    def __init__(self, hits: list[tuple[Chunk, float]] | None = None) -> None:
        self.hits = hits if hits is not None else []
        self.calls: list[tuple[str, int]] = []

    def search_with_scores(self, query: str, k: int = 5):
        self.calls.append((query, k))
        return self.hits


class FakeCaptionStore:
    def __init__(self, hits=None, raises: Exception | None = None) -> None:
        self.hits = hits if hits is not None else []
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def search_with_scores(self, query: str, k: int = 5):
        self.calls.append((query, k))
        if self.raises is not None:
            raise self.raises
        return self.hits


class FakeImageStore:
    def __init__(self, hits=None, raises: Exception | None = None) -> None:
        self.hits = hits if hits is not None else []
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def search_with_scores(self, query: str, k: int = 5):
        self.calls.append((query, k))
        if self.raises is not None:
            raise self.raises
        return self.hits


class FakeReranker:
    """Deterministic stand-in for CrossEncoderReranker.

    Preserves the real contract: never mutates candidates, returns
    RerankedCandidate objects that keep the RRF score and rank separate from
    the reranker score.
    """

    def __init__(self, order: list[str] | None = None, raises: Exception | None = None) -> None:
        self.order = order
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        from mrta.retrieval.reranker import RerankedCandidate, candidate_to_reranker_text

        self.calls.append((query, len(candidates)))
        if self.raises is not None:
            raise self.raises

        if self.order is not None:
            rank_of = {cid: i for i, cid in enumerate(self.order)}
            ordered = sorted(
                enumerate(candidates),
                key=lambda pair: rank_of.get(pair[1].canonical_id, len(rank_of)),
            )
        else:
            ordered = list(enumerate(candidates))

        return [
            RerankedCandidate(
                candidate=cand,
                reranker_score=float(len(candidates) - i),
                reranker_rank=i + 1,
                original_rrf_score=cand.score,
                original_rrf_rank=position + 1,
                reranker_text=candidate_to_reranker_text(cand),
                reranker_text_source="chunk_text",
            )
            for i, (position, cand) in enumerate(ordered[:top_k])
        ]


class FakeVLM:
    def __init__(self, answer: str = "Grounded answer [T1].", raises: Exception | None = None):
        self.answer = answer
        self.raises = raises
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    def generate(self, prompt: str, images: list) -> str:
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        if self.raises is not None and len(self.prompts) == 1:
            raise self.raises
        return self.answer


@pytest.fixture
def text_hits():
    return [
        (_chunk(2, 0, "Attention weights are computed with a softmax."), 0.91),
        (_chunk(6, 1, "Positional encodings use sine and cosine."), 0.72),
    ]


@pytest.fixture
def caption_hits():
    return [
        (_caption_record(4, 1, caption="Diagram of scaled dot-product attention."), 0.83),
    ]


@pytest.fixture
def clip_hits():
    return [
        (_caption_record(4, 1), 0.31),
    ]


# ---------------------------------------------------------------------------
# 2-6: every stage is invoked, using the canonical (not legacy) implementations
# ---------------------------------------------------------------------------


class TestStagesInvoked:
    def test_text_retrieval_invoked(self, text_hits) -> None:
        store = FakeTextStore(text_hits)
        retrieve_multimodal("q", text_store=store)
        assert store.calls == [("q", 20)]

    def test_caption_retrieval_invoked(self, text_hits, caption_hits) -> None:
        caption = FakeCaptionStore(caption_hits)
        retrieve_multimodal("q", text_store=FakeTextStore(text_hits), caption_store=caption)
        assert caption.calls == [("q", 20)]

    def test_clip_retrieval_invoked(self, text_hits, clip_hits) -> None:
        images = FakeImageStore(clip_hits)
        retrieve_multimodal("q", text_store=FakeTextStore(text_hits), image_store=images)
        assert images.calls == [("q", 20)]

    def test_canonical_rrf_is_used_not_legacy(self, text_hits, caption_hits) -> None:
        """The canonical PR4 API must be the one that runs, not evidence_id fusion."""
        with patch(
            "mrta.retrieval.canonical_pipeline.reciprocal_rank_fusion_canonical"
        ) as mock_canonical:
            mock_canonical.return_value = []
            retrieve_multimodal(
                "q",
                text_store=FakeTextStore(text_hits),
                caption_store=FakeCaptionStore(caption_hits),
            )
        assert mock_canonical.call_count == 1
        assert mock_canonical.call_args.kwargs["k"] == 60

    def test_reranker_invoked_over_fused_pool(self, text_hits, caption_hits) -> None:
        reranker = FakeReranker()
        final, diagnostics = retrieve_multimodal(
            "q",
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(caption_hits),
            reranker=reranker,
        )
        assert reranker.calls and reranker.calls[0][0] == "q"
        assert diagnostics.reranker_used is True
        assert final

    def test_pool_depth_and_final_k_are_the_evaluated_values(self, text_hits) -> None:
        store = FakeTextStore(text_hits)
        final, diagnostics = retrieve_multimodal("q", text_store=store, reranker=FakeReranker())
        assert store.calls[0][1] == 20  # candidate pool depth
        assert len(final) <= 5  # final top-k


# ---------------------------------------------------------------------------
# 5, 12, 13: canonical evidence identity
# ---------------------------------------------------------------------------


class TestCanonicalIdentity:
    def test_caption_and_clip_same_figure_collapse_to_one_candidate(
        self, text_hits, caption_hits, clip_hits
    ) -> None:
        final, _ = retrieve_multimodal(
            "q",
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(caption_hits),
            image_store=FakeImageStore(clip_hits),
        )
        figures = [c for c in final if c.candidate.evidence_type == "figure"]
        assert len(figures) == 1
        assert set(figures[0].candidate.modality_sources) == {"caption", "clip"}

    def test_merged_figure_yields_a_single_citation(
        self, text_hits, caption_hits, clip_hits
    ) -> None:
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore(text_hits),
            vlm=FakeVLM(),
            caption_store=FakeCaptionStore(caption_hits),
            image_store=FakeImageStore(clip_hits),
            attach_images=False,
        )
        answer = rag.ask("q")
        assert len(answer.visual_citations) == 1
        assert sorted(answer.visual_citations[0].modality_sources) == ["caption", "clip"]

    def test_text_and_figure_on_same_page_stay_distinct(self) -> None:
        """A chunk and a figure on page 4 must never merge into one candidate."""
        final, _ = retrieve_multimodal(
            "q",
            text_store=FakeTextStore([(_chunk(4, 0, "Body text on page four."), 0.9)]),
            caption_store=FakeCaptionStore([(_caption_record(4, 1, caption="A figure."), 0.8)]),
        )
        kinds = sorted(c.candidate.evidence_type for c in final)
        assert kinds == ["figure", "text"]

    def test_visual_record_and_evidence_record_merge_on_same_figure(self, text_hits) -> None:
        """ImageStore VisualRecords and caption EvidenceRecords must agree on identity."""
        visual = VisualRecord(
            document_id=DOC,
            page=4,
            figure_id="fig_semantic_name",
            figure_index=1,
            image_path="data/figures/f1.png",
        )
        final, _ = retrieve_multimodal(
            "q",
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore([(_caption_record(4, 1, caption="A figure."), 0.8)]),
            image_store=FakeImageStore([(visual, 0.3)]),
        )
        figures = [c for c in final if c.candidate.evidence_type == "figure"]
        assert len(figures) == 1
        assert set(figures[0].candidate.modality_sources) == {"caption", "clip"}


# ---------------------------------------------------------------------------
# 7-10: what reaches generation
# ---------------------------------------------------------------------------


class TestGenerationContext:
    def _ask(self, **kw):
        vlm = FakeVLM()
        rag = CanonicalMultimodalRAG(vlm=vlm, attach_images=False, **kw)
        answer = rag.ask("How is attention computed?")
        return answer, vlm.prompts[0]

    def test_generation_receives_only_final_reranked_evidence(
        self, text_hits, caption_hits
    ) -> None:
        """Evidence the reranker dropped must not appear in the prompt."""
        vlm = FakeVLM()
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore(text_hits),
            vlm=vlm,
            caption_store=FakeCaptionStore(caption_hits),
            reranker=FakeReranker(),
            top_k=1,
            attach_images=False,
        )
        answer = rag.ask("q")
        prompt = vlm.prompts[0]
        # Exactly one candidate survives top_k=1, so the other two retrieved
        # items must be absent from the prompt entirely — not merely unlabelled.
        assert len(answer.text_citations) + len(answer.visual_citations) == 1
        assert "Diagram of scaled dot-product attention." in prompt
        assert "Attention weights are computed with a softmax." not in prompt
        assert "Positional encodings use sine and cosine." not in prompt
        assert "--- TEXT EVIDENCE ---" not in prompt

    def test_text_evidence_formatting(self, text_hits) -> None:
        _, prompt = self._ask(text_store=FakeTextStore(text_hits))
        assert "--- TEXT EVIDENCE ---" in prompt
        assert "[T1]" in prompt
        assert f"Document: {PDF} | Page: 2" in prompt
        assert "Content: Attention weights are computed with a softmax." in prompt

    def test_figure_evidence_formatting(self, text_hits, caption_hits) -> None:
        _, prompt = self._ask(
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(caption_hits),
        )
        assert "--- FIGURE EVIDENCE ---" in prompt
        assert "[F1]" in prompt
        assert "Figure ID: p4_f1" in prompt

    def test_figure_caption_reaches_generation_context(self, text_hits, caption_hits) -> None:
        _, prompt = self._ask(
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(caption_hits),
        )
        assert "Diagram of scaled dot-product attention." in prompt

    def test_text_and_figure_sections_are_separate(self, text_hits, caption_hits) -> None:
        _, prompt = self._ask(
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(caption_hits),
        )
        assert prompt.index("--- TEXT EVIDENCE ---") < prompt.index("--- FIGURE EVIDENCE ---")

    def test_nearby_text_fallback_used_when_no_caption(self, text_hits) -> None:
        record = _caption_record(4, 1, caption=None, nearby_text="Figure 2: attention heads.")
        _, prompt = self._ask(
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore([(record, 0.8)]),
        )
        assert "Figure 2: attention heads." in prompt

    def test_image_path_is_not_presented_as_semantic_content(self, text_hits) -> None:
        """A filesystem path must never be handed to the model as visual information."""
        record = _caption_record(
            4, 1, caption="A diagram.", image_path="data/figures/secret_figure.png"
        )
        _, prompt = self._ask(
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore([(record, 0.8)]),
        )
        assert "secret_figure.png" not in prompt
        assert "A diagram." in prompt


# ---------------------------------------------------------------------------
# 11, 14-16: citation integrity and image-path safety
# ---------------------------------------------------------------------------


class TestCitationIntegrity:
    def test_citations_resolve_only_to_retrieved_evidence(self, text_hits, caption_hits) -> None:
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore(text_hits),
            vlm=FakeVLM(),
            caption_store=FakeCaptionStore(caption_hits),
            attach_images=False,
        )
        answer = rag.ask("q")
        retrieved_pages = {2, 6, 4}
        for citation in answer.text_citations + answer.visual_citations:
            assert citation.source == PDF
            assert citation.document_id == DOC
            assert citation.page in retrieved_pages

    def test_invented_figure_label_is_reported_not_trusted(self, text_hits, caption_hits) -> None:
        views = build_evidence_views(
            retrieve_multimodal(
                "q",
                text_store=FakeTextStore(text_hits),
                caption_store=FakeCaptionStore(caption_hits),
            )[0]
        )
        result = verify_answer_citations("As shown in [F1] and [F9], see [T1].", views)
        assert "[F9]" in result.unknown
        assert result.is_clean is False
        assert set(result.referenced) <= {v.label for v in views}

    def test_invented_labels_never_become_citations(self, text_hits) -> None:
        """Citations come from retrieved evidence, so model text cannot add any."""
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore(text_hits),
            vlm=FakeVLM(answer="See [F4] on page 99 of other.pdf and [T7]."),
            attach_images=False,
        )
        answer = rag.ask("q")
        assert answer.visual_citations == []
        labels = {c.label for c in answer.text_citations}
        assert "[T7]" not in labels
        assert all(c.page != 99 for c in answer.text_citations)

    def test_nonexistent_image_path_is_not_emitted(self, text_hits) -> None:
        record = _caption_record(4, 1, caption="A.", image_path="data/does/not/exist.png")
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore(text_hits),
            vlm=FakeVLM(),
            caption_store=FakeCaptionStore([(record, 0.8)]),
            attach_images=False,
        )
        answer = rag.ask("q")
        assert answer.visual_citations[0].image_path is None

    def test_absolute_and_escaping_paths_are_rejected(self) -> None:
        assert safe_image_path("/etc/passwd") is None
        assert safe_image_path("../../etc/passwd") is None
        assert safe_image_path("secrets/key.png") is None
        assert safe_image_path(None) is None

    def test_valid_image_metadata_survives_to_the_response(self, tmp_path, text_hits) -> None:
        import os

        cwd = os.getcwd()
        asset = tmp_path / "data" / "figures"
        asset.mkdir(parents=True)
        (asset / "f1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        os.chdir(tmp_path)
        try:
            record = _caption_record(4, 1, caption="A diagram.", image_path="data/figures/f1.png")
            rag = CanonicalMultimodalRAG(
                text_store=FakeTextStore(text_hits),
                vlm=FakeVLM(),
                caption_store=FakeCaptionStore([(record, 0.8)]),
                attach_images=False,
            )
            answer = rag.ask("q")
        finally:
            os.chdir(cwd)
        citation = answer.visual_citations[0]
        assert citation.image_path == "data/figures/f1.png"
        assert citation.figure_id == "p4_f1"
        assert citation.caption == "A diagram."


# ---------------------------------------------------------------------------
# 20-22: degradation and failure isolation
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    def test_empty_retrieval_handled_cleanly(self) -> None:
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore([]), vlm=FakeVLM(), attach_images=False
        )
        answer = rag.ask("q")
        assert answer.text_citations == []
        assert answer.visual_citations == []
        assert isinstance(answer.answer, str)

    def test_empty_retrieval_returns_no_candidates(self) -> None:
        final, diagnostics = retrieve_multimodal("q", text_store=FakeTextStore([]))
        assert final == []
        assert diagnostics.stream_sizes["text"] == 0

    def test_caption_stream_failure_degrades_without_killing_query(
        self, text_hits, clip_hits
    ) -> None:
        final, diagnostics = retrieve_multimodal(
            "q",
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(raises=RuntimeError("caption index corrupt")),
            image_store=FakeImageStore(clip_hits),
        )
        assert final
        assert "caption" in diagnostics.degraded_streams
        assert "caption index corrupt" in diagnostics.degraded_streams["caption"]

    def test_clip_stream_failure_degrades_without_killing_query(
        self, text_hits, caption_hits
    ) -> None:
        final, diagnostics = retrieve_multimodal(
            "q",
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(caption_hits),
            image_store=FakeImageStore(raises=RuntimeError("clip index missing")),
        )
        assert final
        assert "clip" in diagnostics.degraded_streams

    def test_absent_visual_streams_leave_text_path_working(self, text_hits) -> None:
        final, diagnostics = retrieve_multimodal("q", text_store=FakeTextStore(text_hits))
        assert len(final) == 2
        assert "caption" not in diagnostics.stream_sizes
        assert "clip" not in diagnostics.stream_sizes

    def test_reranker_failure_falls_back_to_rrf_ordering(self, text_hits, caption_hits) -> None:
        final, diagnostics = retrieve_multimodal(
            "q",
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(caption_hits),
            reranker=FakeReranker(raises=RuntimeError("weights unavailable")),
        )
        assert final
        assert diagnostics.reranker_used is False
        assert "reranker" in diagnostics.degraded_streams
        assert all(c.reranker_text_source == "not_reranked" for c in final)

    def test_absent_reranker_returns_rrf_ordering(self, text_hits) -> None:
        final, diagnostics = retrieve_multimodal("q", text_store=FakeTextStore(text_hits))
        assert diagnostics.reranker_used is False
        assert [c.reranker_rank for c in final] == [1, 2]

    def test_generation_falls_back_when_images_rejected(self, text_hits, caption_hits) -> None:
        vlm = FakeVLM(raises=LLMError("model has no vision support"))
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore(text_hits),
            vlm=vlm,
            caption_store=FakeCaptionStore(caption_hits),
        )
        answer = rag.ask("q")
        assert answer.retrieval_mode == "text_only"
        assert len(vlm.prompts) == 2  # retried without images


# ---------------------------------------------------------------------------
# Score provenance (PR4/PR5 invariants must survive integration)
# ---------------------------------------------------------------------------


class TestScoreProvenance:
    def test_rrf_and_reranker_scores_stay_separate(self, text_hits, caption_hits) -> None:
        final, _ = retrieve_multimodal(
            "q",
            text_store=FakeTextStore(text_hits),
            caption_store=FakeCaptionStore(caption_hits),
            reranker=FakeReranker(),
        )
        for candidate in final:
            assert candidate.candidate.score == candidate.original_rrf_score
            assert candidate.original_rrf_rank >= 1
            assert candidate.reranker_rank >= 1

    def test_diagnostics_describe_the_same_run_as_the_answer(self, text_hits, caption_hits) -> None:
        """ask_with_diagnostics must not re-run retrieval behind the answer."""
        text_store = FakeTextStore(text_hits)
        caption_store = FakeCaptionStore(caption_hits)
        vlm = FakeVLM()
        rag = CanonicalMultimodalRAG(
            text_store=text_store,
            vlm=vlm,
            caption_store=caption_store,
            reranker=FakeReranker(),
            attach_images=False,
        )
        answer, diagnostics, verification = rag.ask_with_diagnostics("q")
        assert len(text_store.calls) == 1
        assert len(caption_store.calls) == 1
        assert len(vlm.prompts) == 1
        assert diagnostics.reranker_used is True
        assert verification.unknown == []
        assert answer.text_citations

    def test_ranking_provenance_reaches_citations(self, text_hits, caption_hits) -> None:
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore(text_hits),
            vlm=FakeVLM(),
            caption_store=FakeCaptionStore(caption_hits),
            reranker=FakeReranker(),
            attach_images=False,
        )
        answer = rag.ask("q")
        for citation in answer.text_citations + answer.visual_citations:
            assert citation.rrf_rank is not None
            assert citation.reranker_rank is not None


# ---------------------------------------------------------------------------
# 23-24: determinism and prompt hygiene
# ---------------------------------------------------------------------------


class TestDeterminismAndHygiene:
    def test_repeated_runs_are_deterministic(self, text_hits, caption_hits, clip_hits) -> None:
        results = []
        for _ in range(3):
            final, _ = retrieve_multimodal(
                "q",
                text_store=FakeTextStore(text_hits),
                caption_store=FakeCaptionStore(caption_hits),
                image_store=FakeImageStore(clip_hits),
                reranker=FakeReranker(),
            )
            results.append([c.candidate.canonical_id for c in final])
        assert results[0] == results[1] == results[2]

    def test_no_benchmark_ground_truth_reaches_the_prompt(self, text_hits) -> None:
        """Benchmark annotation keys must not appear in generation context."""
        record = _caption_record(4, 1, caption="A diagram.")
        vlm = FakeVLM()
        rag = CanonicalMultimodalRAG(
            text_store=FakeTextStore(text_hits),
            vlm=vlm,
            caption_store=FakeCaptionStore([(record, 0.8)]),
            attach_images=False,
        )
        rag.ask("q")
        prompt = vlm.prompts[0]
        for leaked in (
            "expected_evidence",
            "source_note",
            "retrieval_challenge",
            "difficulty",
            "target_evidence",
        ):
            assert leaked not in prompt

    def test_production_figure_ids_are_derived_not_semantic(self, text_hits) -> None:
        """Production must not depend on the benchmark's semantic figure names."""
        visual = VisualRecord(
            document_id=DOC,
            page=4,
            figure_id="fig_transformer_arch",
            figure_index=1,
            image_path="data/figures/f1.png",
        )
        final, _ = retrieve_multimodal(
            "q", text_store=FakeTextStore(text_hits), image_store=FakeImageStore([(visual, 0.5)])
        )
        figures = [c for c in final if c.candidate.evidence_type == "figure"]
        assert figures[0].candidate.figure_id == "p4_f1"


# ---------------------------------------------------------------------------
# 1, 17-19, 25: backward compatibility
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.search_with_scores.return_value = [(_chunk(1, 0, "Attention is all you need."), 0.9)]
    return store


@pytest.fixture
def api_client(mock_store):
    llm = MagicMock()
    llm.model = "test-model"
    llm.chat.return_value = "A grounded answer."
    app.dependency_overrides[get_store] = lambda: mock_store
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_retriever] = lambda: None
    app.dependency_overrides[get_vlm] = lambda: None
    app.dependency_overrides[get_canonical_stack] = lambda: None
    from fastapi.testclient import TestClient

    with (
        patch("apps.api.main.Embedder"),
        patch("apps.api.main.VectorStore"),
        patch("apps.api.main.LLMClient"),
        # The multimodal constructors must be patched as well, matching
        # tests/unit/test_api.py. Without this the lifespan builds a real CLIP
        # model: slow, and on macOS it segfaults outright once another test has
        # already initialized FAISS's OpenMP runtime (see CLIPEmbedder.warmup).
        patch("apps.api.main._CLIPEmbedder", create=True),
        patch("apps.api.main._VisualVectorStore", create=True),
        patch("apps.api.main._MultimodalRetriever", create=True),
        patch("apps.api.main._VLMClient", create=True),
        TestClient(app) as client,
    ):
        yield client
    app.dependency_overrides.clear()


class TestBackwardCompatibility:
    def test_text_only_query_path_still_works(self, api_client) -> None:
        response = api_client.post("/ask", json={"question": "What is attention?"})
        assert response.status_code == 200

    def test_legacy_response_fields_all_present(self, api_client) -> None:
        body = api_client.post("/ask", json={"question": "What is attention?"}).json()
        for field in ("answer", "sources", "latency_s"):
            assert field in body
        assert isinstance(body["sources"], list)
        for field in ("page", "source", "chunk_id", "preview"):
            assert field in body["sources"][0]

    def test_api_routes_unchanged(self, api_client) -> None:
        paths = set(app.openapi()["paths"])
        assert {"/ask", "/upload", "/documents", "/figures", "/health"} <= paths

    def test_health_route_still_ok(self, api_client) -> None:
        assert api_client.get("/health").json() == {"status": "ok"}

    def test_only_question_is_required(self, api_client) -> None:
        assert app.openapi()["components"]["schemas"]["AskRequest"]["required"] == ["question"]

    def test_multimodal_unavailable_still_returns_503(self, api_client) -> None:
        response = api_client.post(
            "/ask", json={"question": "What is attention?", "retrieval_mode": "multimodal"}
        )
        assert response.status_code == 503

    def test_additive_visual_source_fields_exist(self) -> None:
        properties = app.openapi()["components"]["schemas"]["VisualSource"]["properties"]
        for field in ("label", "page", "source", "figure_index", "modality"):
            assert field in properties
        for field in ("document_id", "figure_id", "image_path", "caption", "modality_sources"):
            assert field in properties

    def test_ingest_script_entry_point_intact(self) -> None:
        """The repository's only ingestion command must keep working."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/ingest.py"], capture_output=True, text=True
        )
        assert result.returncode == 1
        assert "Usage: python scripts/ingest.py" in result.stdout

    def test_legacy_multimodal_retriever_still_importable(self) -> None:
        """PR6 must not remove the Stage-7 path that teaching modes depend on."""
        from mrta.generation.multimodal_rag import MultimodalRAG
        from mrta.retrieval.multimodal_retriever import MultimodalRetriever

        assert MultimodalRetriever is not None
        assert MultimodalRAG.VALID_TEACHING_MODES

    def test_legacy_fusion_api_untouched(self) -> None:
        from mrta.retrieval.fusion import FusedResult, reciprocal_rank_fusion

        assert reciprocal_rank_fusion is not None
        assert FusedResult is not None
