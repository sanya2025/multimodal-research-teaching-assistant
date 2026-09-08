"""mrta.eval.types — canonical evidence types for retrieval evaluation.

CanonicalEvidence is a stable semantic identity for one piece of evidence
(a text page or a figure) that remains stable across chunk-size changes,
embedding updates, or chunk UUID regeneration.

Matching rule:
  - If either evidence has a figure_id set, all three fields must match.
  - If both have figure_id=None, only (document_id, page_number) must match.

This means a retrieved text chunk on page 3 matches a target of
CanonicalEvidence(doc, page=3, figure_id=None), but NOT a target of
CanonicalEvidence(doc, page=3, figure_id="fig_transformer_arch").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalEvidence:
    """Stable semantic identity: (document_id, page_number, figure_id)."""

    document_id: str
    page_number: int
    figure_id: str | None = None

    def matches(self, other: CanonicalEvidence) -> bool:
        """Return True if self and other refer to the same evidence unit."""
        doc_match = self.document_id == other.document_id
        page_match = self.page_number == other.page_number
        if self.figure_id is not None or other.figure_id is not None:
            return doc_match and page_match and self.figure_id == other.figure_id
        return doc_match and page_match

    def key(self) -> tuple[str, int, str | None]:
        """Hashable deduplication key."""
        return (self.document_id, self.page_number, self.figure_id)


@dataclass
class RetrievedCandidate:
    """One retrieved result with its canonical evidence mapping and retrieval score."""

    candidate_id: str
    evidence: CanonicalEvidence
    score: float
    rank: int
