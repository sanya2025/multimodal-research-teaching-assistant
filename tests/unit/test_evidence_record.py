"""Tests for EvidenceRecord schema and FigureRecord.to_evidence_record()."""

from __future__ import annotations

import pytest

from mrta.core.schemas import Chunk, EvidenceRecord, FigureRecord


class TestEvidenceRecordConstruction:
    def test_text_modality(self) -> None:
        rec = EvidenceRecord(
            evidence_id="doc_p1_c0",
            doc_id="doc",
            source="paper.pdf",
            page=1,
            modality="text",
            text="Self-attention is the key mechanism.",
        )
        assert rec.modality == "text"
        assert rec.text == "Self-attention is the key mechanism."

    def test_image_modality(self) -> None:
        rec = EvidenceRecord(
            evidence_id="doc_p3_f1",
            doc_id="doc",
            source="paper.pdf",
            page=3,
            modality="image",
            figure_index=1,
            image_bytes=b"\x89PNG",
        )
        assert rec.modality == "image"
        assert rec.figure_index == 1

    def test_page_modality(self) -> None:
        rec = EvidenceRecord(
            evidence_id="doc_p2_page",
            doc_id="doc",
            source="paper.pdf",
            page=2,
            modality="page",
            image_bytes=b"\x89PNG",
        )
        assert rec.modality == "page"

    def test_invalid_modality_raises(self) -> None:
        with pytest.raises(Exception):
            EvidenceRecord(
                evidence_id="x",
                doc_id="x",
                source="x.pdf",
                page=1,
                modality="audio",  # type: ignore[arg-type]
            )

    def test_optional_fields_default_to_none(self) -> None:
        rec = EvidenceRecord(
            evidence_id="x",
            doc_id="x",
            source="x.pdf",
            page=1,
            modality="text",
        )
        assert rec.figure_index is None
        assert rec.bbox is None
        assert rec.image_bytes is None
        assert rec.caption is None
        assert rec.detailed_description is None
        assert rec.visual_type is None


class TestEvidenceRecordSerialisation:
    def test_round_trip_json(self) -> None:
        rec = EvidenceRecord(
            evidence_id="doc_p1_f2",
            doc_id="doc",
            source="paper.pdf",
            page=1,
            modality="image",
            figure_index=2,
            caption="Encoder-decoder architecture.",
        )
        serialised = rec.model_dump_json()
        restored = EvidenceRecord.model_validate_json(serialised)
        assert restored.evidence_id == rec.evidence_id
        assert restored.caption == rec.caption

    def test_retrieval_score_excluded_from_json(self) -> None:
        rec = EvidenceRecord(
            evidence_id="x",
            doc_id="x",
            source="x.pdf",
            page=1,
            modality="text",
            retrieval_score=0.92,
        )
        import json

        data = json.loads(rec.model_dump_json())
        assert "retrieval_score" not in data


class TestRetrievalText:
    def test_prefers_caption(self) -> None:
        rec = EvidenceRecord(
            evidence_id="x",
            doc_id="x",
            source="x.pdf",
            page=1,
            modality="image",
            caption="A figure caption.",
            detailed_description="A longer description.",
        )
        assert rec.retrieval_text() == "A figure caption."

    def test_falls_back_to_detailed_description(self) -> None:
        rec = EvidenceRecord(
            evidence_id="x",
            doc_id="x",
            source="x.pdf",
            page=1,
            modality="image",
            detailed_description="A longer description.",
        )
        assert rec.retrieval_text() == "A longer description."

    def test_falls_back_to_nearby_text(self) -> None:
        rec = EvidenceRecord(
            evidence_id="x",
            doc_id="x",
            source="x.pdf",
            page=1,
            modality="image",
            nearby_text="surrounding page text",
        )
        assert rec.retrieval_text() == "surrounding page text"

    def test_empty_string_when_no_text(self) -> None:
        rec = EvidenceRecord(
            evidence_id="x",
            doc_id="x",
            source="x.pdf",
            page=1,
            modality="image",
        )
        assert rec.retrieval_text() == ""


class TestFromChunk:
    def test_wraps_chunk_correctly(self) -> None:
        chunk = Chunk(
            chunk_id="doc_p5_c3",
            doc_id="doc",
            source="paper.pdf",
            page=5,
            text="The model uses multi-head attention.",
        )
        rec = EvidenceRecord.from_chunk(chunk)
        assert rec.evidence_id == chunk.chunk_id
        assert rec.modality == "text"
        assert rec.text == chunk.text
        assert rec.page == 5


class TestFigureRecordToEvidenceRecord:
    def test_conversion_preserves_metadata(self) -> None:
        fig = FigureRecord(
            doc_id="doc",
            source="paper.pdf",
            page=4,
            figure_index=2,
            image_bytes=b"\x89PNG",
            width=300,
            height=200,
            bbox=(10.0, 20.0, 310.0, 220.0),
            nearby_text="Figure 2: Attention weights.",
        )
        rec = fig.to_evidence_record()
        assert rec.modality == "image"
        assert rec.figure_index == 2
        assert rec.page == 4
        assert rec.bbox == (10.0, 20.0, 310.0, 220.0)
        assert rec.nearby_text == "Figure 2: Attention weights."
        assert rec.evidence_id == "doc_p4_f2"

    def test_evidence_id_format(self) -> None:
        fig = FigureRecord(
            doc_id="mydoc",
            source="paper.pdf",
            page=7,
            figure_index=3,
            image_bytes=b"x",
        )
        assert fig.to_evidence_record().evidence_id == "mydoc_p7_f3"
