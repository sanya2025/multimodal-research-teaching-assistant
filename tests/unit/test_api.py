"""Unit tests for the FastAPI backend (apps/api/).

Uses TestClient with dependency_overrides so no live Ollama or embedding model
is needed for routing logic. The lifespan still runs and loads the real Embedder
(MiniLM in test env) and an empty VectorStore — but the routes receive mock objects
via the overrides, so results are deterministic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from apps.api.deps import get_llm, get_retriever, get_store, get_vlm
from apps.api.main import app
from fastapi.testclient import TestClient

from mrta.core.schemas import Chunk, MultimodalAnswer, MultimodalCitation

FAKE_CHUNKS = [
    Chunk(
        chunk_id="doc_abc_p1_c0",
        doc_id="doc_abc",
        source="attention.pdf",
        page=1,
        text="Attention is all you need.",
    ),
    Chunk(
        chunk_id="doc_abc_p2_c0",
        doc_id="doc_abc",
        source="attention.pdf",
        page=2,
        text="Multi-head attention allows joint attention over different representation subspaces.",
    ),
]

MOCK_ANSWER = "According to [page 1], attention is the core mechanism."


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.search_with_scores.return_value = [(c, 0.9) for c in FAKE_CHUNKS]
    store._chunks = FAKE_CHUNKS
    return store


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = MOCK_ANSWER
    return llm


_LIFESPAN_PATCHES = (
    "apps.api.main.Embedder",
    "apps.api.main.VectorStore",
    "apps.api.main.LLMClient",
)


@pytest.fixture
def client(mock_store: MagicMock, mock_llm: MagicMock):
    app.dependency_overrides[get_store] = lambda: mock_store
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_retriever] = lambda: None  # multimodal unavailable
    app.dependency_overrides[get_vlm] = lambda: None
    with (
        patch("apps.api.main.Embedder"),
        patch("apps.api.main.VectorStore"),
        patch("apps.api.main.LLMClient"),
        patch("apps.api.main._CLIPEmbedder", create=True),
        patch("apps.api.main._VisualVectorStore", create=True),
        patch("apps.api.main._MultimodalRetriever", create=True),
        patch("apps.api.main._VLMClient", create=True),
        TestClient(app) as c,
    ):
        yield c
    app.dependency_overrides.clear()


_MM_ANSWER = MultimodalAnswer(
    answer="Multimodal answer.",
    text_citations=[
        MultimodalCitation(
            label="[T1]", evidence_id="t1", modality="text", source="attention.pdf", page=1
        )
    ],
    visual_citations=[
        MultimodalCitation(
            label="[V1]",
            evidence_id="v1",
            modality="image",
            source="attention.pdf",
            page=2,
            figure_index=1,
        )
    ],
    retrieval_mode="multimodal",
    latency_s=0.5,
)


@pytest.fixture
def mm_client(mock_store: MagicMock, mock_llm: MagicMock):
    """TestClient with multimodal retriever wired and MultimodalRAG mocked."""
    mock_retriever = MagicMock()
    app.dependency_overrides[get_store] = lambda: mock_store
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_vlm] = lambda: MagicMock()
    with (
        patch("apps.api.main.Embedder"),
        patch("apps.api.main.VectorStore"),
        patch("apps.api.main.LLMClient"),
        patch("apps.api.main._CLIPEmbedder", create=True),
        patch("apps.api.main._VisualVectorStore", create=True),
        patch("apps.api.main._MultimodalRetriever", create=True),
        patch("apps.api.main._VLMClient", create=True),
        patch("apps.api.routers.ask.MultimodalRAG") as MockRAG,
        TestClient(app) as c,
    ):
        MockRAG.return_value.ask.return_value = _MM_ANSWER
        yield c
    app.dependency_overrides.clear()


class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_returns_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.json() == {"status": "ok"}


class TestAsk:
    def test_valid_payload_returns_200(self, client: TestClient) -> None:
        r = client.post("/ask", json={"question": "What is attention?", "top_k": 3})
        assert r.status_code == 200

    def test_response_has_answer_and_sources(self, client: TestClient) -> None:
        r = client.post("/ask", json={"question": "What is attention?", "top_k": 3})
        data = r.json()
        assert "answer" in data
        assert "sources" in data

    def test_short_question_returns_422(self, client: TestClient) -> None:
        r = client.post("/ask", json={"question": "ab"})
        assert r.status_code == 422

    def test_sources_contain_page_and_chunk_id(self, client: TestClient) -> None:
        r = client.post("/ask", json={"question": "What is attention?", "top_k": 2})
        sources = r.json()["sources"]
        assert len(sources) > 0
        assert "page" in sources[0]
        assert "chunk_id" in sources[0]


class TestDocuments:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/documents")
        assert r.status_code == 200

    def test_returns_list(self, client: TestClient) -> None:
        r = client.get("/documents")
        assert isinstance(r.json(), list)

    def test_returns_document_info_shape(self, client: TestClient) -> None:
        r = client.get("/documents")
        doc = r.json()[0]
        assert "doc_id" in doc
        assert "source" in doc
        assert "n_pages" in doc
        assert "n_chunks" in doc

    def test_aggregates_chunks_by_doc_id(self, client: TestClient) -> None:
        r = client.get("/documents")
        # FAKE_CHUNKS both belong to doc_abc — expect one DocumentInfo
        assert len(r.json()) == 1
        assert r.json()[0]["doc_id"] == "doc_abc"


class TestUpload:
    def test_pdf_upload_returns_200(self, client: TestClient) -> None:
        pdf_path = Path("tests/fixtures/sample.pdf")
        with patch("apps.api.routers.upload.chunk_pdf") as mock_chunk:
            mock_chunk.return_value = FAKE_CHUNKS
            with pdf_path.open("rb") as f:
                r = client.post("/upload", files={"file": ("sample.pdf", f, "application/pdf")})
        assert r.status_code == 200

    def test_pdf_upload_returns_expected_fields(self, client: TestClient) -> None:
        pdf_path = Path("tests/fixtures/sample.pdf")
        with patch("apps.api.routers.upload.chunk_pdf") as mock_chunk:
            mock_chunk.return_value = FAKE_CHUNKS
            with pdf_path.open("rb") as f:
                r = client.post("/upload", files={"file": ("sample.pdf", f, "application/pdf")})
        data = r.json()
        assert "doc_id" in data
        assert "n_pages" in data
        assert "n_chunks" in data
        assert data["n_chunks"] == len(FAKE_CHUNKS)

    def test_non_pdf_returns_400(self, client: TestClient) -> None:
        r = client.post("/upload", files={"file": ("note.txt", b"hello", "text/plain")})
        assert r.status_code == 400

    def test_oversized_file_returns_413(self, client: TestClient) -> None:
        big = b"%PDF" + b"x" * (20 * 1024 * 1024 + 1)
        r = client.post("/upload", files={"file": ("big.pdf", big, "application/pdf")})
        assert r.status_code == 413

    def test_non_pdf_magic_bytes_returns_415(self, client: TestClient) -> None:
        r = client.post(
            "/upload",
            files={"file": ("fake.pdf", b"PK\x03\x04not-a-pdf", "application/pdf")},
        )
        assert r.status_code == 415

    def test_path_traversal_filename_is_sanitised(self, client: TestClient) -> None:
        pdf_path = Path("tests/fixtures/sample.pdf")
        with patch("apps.api.routers.upload.chunk_pdf") as mock_chunk:
            mock_chunk.return_value = FAKE_CHUNKS
            with pdf_path.open("rb") as f:
                r = client.post(
                    "/upload",
                    files={"file": ("../../evil.pdf", f, "application/pdf")},
                )
        assert r.status_code == 200
        assert r.json()["source"] == "evil.pdf"

    def test_malformed_pdf_returns_422(self, client: TestClient) -> None:
        with patch("apps.api.routers.upload.load_pdf") as mock_load:
            from mrta.core.exceptions import IngestionError

            mock_load.side_effect = IngestionError("Cannot open PDF")
            r = client.post(
                "/upload",
                files={"file": ("broken.pdf", b"%PDF-broken", "application/pdf")},
            )
        assert r.status_code == 422
        assert r.json()["code"] == "malformed_pdf"

    def test_duplicate_upload_returns_already_indexed(self, client: TestClient) -> None:
        # FAKE_CHUNKS use source="attention.pdf" — uploading that name hits the guard
        pdf_path = Path("tests/fixtures/sample.pdf")
        with pdf_path.open("rb") as f:
            r = client.post(
                "/upload",
                files={"file": ("attention.pdf", f, "application/pdf")},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["already_indexed"] is True
        assert data["source"] == "attention.pdf"
        assert data["n_chunks"] == len(FAKE_CHUNKS)


class TestAskMultimodal:
    """POST /ask with retrieval_mode='multimodal'."""

    def test_multimodal_returns_200(self, mm_client: TestClient) -> None:
        r = mm_client.post(
            "/ask", json={"question": "What is attention?", "retrieval_mode": "multimodal"}
        )
        assert r.status_code == 200

    def test_multimodal_response_has_answer(self, mm_client: TestClient) -> None:
        r = mm_client.post(
            "/ask", json={"question": "What is attention?", "retrieval_mode": "multimodal"}
        )
        assert r.json()["answer"] == "Multimodal answer."

    def test_multimodal_response_retrieval_mode_field(self, mm_client: TestClient) -> None:
        r = mm_client.post(
            "/ask", json={"question": "What is attention?", "retrieval_mode": "multimodal"}
        )
        assert r.json()["retrieval_mode"] == "multimodal"

    def test_multimodal_response_has_visual_sources(self, mm_client: TestClient) -> None:
        r = mm_client.post(
            "/ask", json={"question": "What is attention?", "retrieval_mode": "multimodal"}
        )
        visual = r.json()["visual_sources"]
        assert len(visual) == 1
        assert visual[0]["label"] == "[V1]"
        assert visual[0]["page"] == 2
        assert visual[0]["figure_index"] == 1

    def test_multimodal_response_has_text_sources(self, mm_client: TestClient) -> None:
        r = mm_client.post(
            "/ask", json={"question": "What is attention?", "retrieval_mode": "multimodal"}
        )
        sources = r.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["page"] == 1

    def test_teaching_mode_accepted_in_request(self, mm_client: TestClient) -> None:
        r = mm_client.post(
            "/ask",
            json={
                "question": "What is attention?",
                "retrieval_mode": "multimodal",
                "teaching_mode": "explain",
            },
        )
        assert r.status_code == 200

    def test_all_teaching_modes_accepted(self, mm_client: TestClient) -> None:
        modes = ["explain", "socratic", "quiz", "compare", "visual_evidence"]
        for mode in modes:
            r = mm_client.post(
                "/ask",
                json={
                    "question": "What is attention?",
                    "retrieval_mode": "multimodal",
                    "teaching_mode": mode,
                },
            )
            assert r.status_code == 200, f"mode={mode} returned {r.status_code}"

    def test_multimodal_unavailable_returns_503(self, client: TestClient) -> None:
        # client fixture has get_retriever returning None (default app.state)
        r = client.post(
            "/ask", json={"question": "What is attention?", "retrieval_mode": "multimodal"}
        )
        assert r.status_code == 503

    def test_text_mode_default_backward_compat(self, client: TestClient) -> None:
        # No retrieval_mode field → defaults to text → should not 503
        r = client.post("/ask", json={"question": "What is attention?"})
        assert r.status_code == 200

    def test_response_visual_sources_empty_for_text_mode(self, client: TestClient) -> None:
        r = client.post("/ask", json={"question": "What is attention?"})
        assert r.json().get("visual_sources", []) == []
