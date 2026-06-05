"""mrta.observability.logging — JSONL run logger."""

from __future__ import annotations

import json
from pathlib import Path

from mrta.core.config import settings
from mrta.core.schemas import Chunk


class StructuredLogger:
    """Appends one JSON line per RAG run to settings.log_file."""

    def log_run(
        self,
        question: str,
        answer: str,
        sources: list[Chunk],
        latency_s: float,
    ) -> None:
        """Append one JSON line to settings.log_file."""
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "question": question,
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "latency_s": latency_s,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
