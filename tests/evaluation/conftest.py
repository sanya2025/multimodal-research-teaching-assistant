from __future__ import annotations

from pathlib import Path

import pytest
import yaml

GOLDEN_QA_PATH = Path(__file__).parents[2] / "data" / "golden_qa.yaml"


def load_golden_qa(path: Path = GOLDEN_QA_PATH) -> list[dict]:
    """Load golden QA items from YAML, normalising optional fields to safe defaults."""
    data = yaml.safe_load(path.read_text())
    return [
        {
            "id": q["id"],
            "question": q["question"],
            "expected_answer": q.get("expected_answer", ""),
            "expected_documents": q.get("expected_documents", []),
            "expected_keywords": q.get("expected_keywords", []),
            "source": q.get("source", ""),
        }
        for q in data.get("questions", [])
    ]


@pytest.fixture(scope="session")
def golden_qa() -> list[dict]:
    return load_golden_qa()
