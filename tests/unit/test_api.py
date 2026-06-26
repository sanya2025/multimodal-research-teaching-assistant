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
from apps.api.deps import get_llm, get_store
from apps.api.main import app
from fastapi.testclient import TestClient

from mrta.core.schemas import Chunk

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


@pytest.fixture
def client(mock_store: MagicMock, mock_llm: MagicMock):
    app.dependency_overrides[get_store] = lambda: mock_store
    app.dependency_overrides[get_llm] = lambda: mock_llm
    with (
        patch("apps.api.main.Embedder"),  # prevents HuggingFace download in lifespan
        patch("apps.api.main.VectorStore"),  # prevents FAISS IndexFlatIP construction
        patch("apps.api.main.LLMClient"),  # prevents Ollama connection in lifespan
        TestClient(app) as c,
    ):
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
