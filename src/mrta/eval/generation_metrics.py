"""mrta.eval.generation_metrics — grounded-generation metrics over canonical evidence.

Evaluation-only. Scores an answer and its citations against the frozen benchmark's
canonical targets, using the same ``CanonicalEvidence`` semantics as PR4/PR5 so
retrieval and generation numbers describe the same notion of "correct evidence".

Citations are resolved to canonical evidence before scoring — never compared as
strings. A citation label is an application-assigned handle ("[T1]", "[F1]"),
so string equality would measure label formatting rather than whether the answer
pointed at the right page or figure.

Three distinct failure modes, deliberately not collapsed
-------------------------------------------------------
``validity``  — did the citation resolve to evidence the generator was actually
                given? An unresolvable citation is fabricated provenance.
``relevance`` — of the evidence cited, how much was the *expected* evidence?
                Measured by precision; a valid-but-irrelevant citation is a
                precision loss, not a hallucination.
``coverage``  — was the expected evidence cited at all, and for hybrid queries
                were both the text and figure targets cited?

Conflating these hides which stage is at fault, which is the whole point of the
PR7 ablation.

On "faithfulness"
-----------------
Nothing here is named ``faithfulness`` or ``hallucination_rate``. Every grounding
signal in this module is *lexical*: it asks whether an answer's tokens appear in
the supplied context, which underestimates correct paraphrase and overestimates
coincidental word overlap. The names say ``support_proxy`` because that is what
they measure. They are useful as deterministic regression signals, not as a
semantic verdict.

(``mrta.evaluation.metrics`` has older functions named ``faithfulness`` and
``hallucination_rate`` that are also pure lexical overlap. They are Stage-7 API
and left untouched; they are not reused here, and their names should not be
taken as a stronger claim than this module's.)
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from mrta.eval.types import CanonicalEvidence

# Tokens shorter than this carry little grounding signal ("the", "is", "a") and
# would inflate any overlap score. Matches the threshold the Stage-7 metric uses.
_MIN_SUPPORT_TOKEN_LEN = 4

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# Numbers and measurements are the claims most worth checking, so they are kept
# even when shorter than the token-length floor.
_NUMERIC = re.compile(r"^\d+(?:\.\d+)?$")


def _dedupe(evidence: Iterable[CanonicalEvidence]) -> list[CanonicalEvidence]:
    """Distinct canonical evidence, preserving first-seen order.

    Deduplication happens before scoring because citing the same figure twice is
    one piece of evidence, not two. Without this, a repeated citation would
    inflate or deflate precision depending on whether it was correct.
    """
    seen: set[tuple] = set()
    out: list[CanonicalEvidence] = []
    for item in evidence:
        key = item.key()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _matches_any(candidate: CanonicalEvidence, targets: Sequence[CanonicalEvidence]) -> bool:
    return any(target.matches(candidate) for target in targets)


# ---------------------------------------------------------------------------
# Citation precision / recall / F1
# ---------------------------------------------------------------------------


def citation_precision_recall(
    response_citations: Sequence[CanonicalEvidence],
    target_evidence: Sequence[CanonicalEvidence],
) -> dict[str, float]:
    """Citation precision, recall and F1 over deduplicated canonical evidence.

    precision = correct cited targets / all cited canonical evidence
    recall    = correct cited targets / all expected canonical targets

    Recall counts *distinct targets matched*, not matching citations, so citing
    one target from two angles cannot make recall exceed 1.0.

    Edge cases, chosen so an empty case never silently looks like success:
      - no expected targets and no citations -> all 1.0 (nothing was required)
      - no expected targets but citations present -> precision 0.0 (everything
        cited is unexpected), recall 1.0
      - expected targets but no citations -> precision 1.0 (vacuous: nothing
        wrong was claimed), recall 0.0
    """
    cited = _dedupe(response_citations)
    targets = _dedupe(target_evidence)

    if not targets:
        precision = 1.0 if not cited else 0.0
        return _with_f1({"citation_precision": precision, "citation_recall": 1.0})

    if not cited:
        return _with_f1({"citation_precision": 1.0, "citation_recall": 0.0})

    correct = [c for c in cited if _matches_any(c, targets)]
    matched_targets = [t for t in targets if any(t.matches(c) for c in cited)]

    return _with_f1(
        {
            "citation_precision": len(correct) / len(cited),
            "citation_recall": len(matched_targets) / len(targets),
        }
    )


def _with_f1(scores: dict[str, float]) -> dict[str, float]:
    precision, recall = scores["citation_precision"], scores["citation_recall"]
    denominator = precision + recall
    scores["citation_f1"] = 0.0 if denominator == 0 else 2 * precision * recall / denominator
    return {k: round(v, 6) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Citation validity (distinct from relevance)
# ---------------------------------------------------------------------------


@dataclass
class CitationValidity:
    """Whether cited evidence was actually supplied to the generator.

    ``invalid`` counts citations that resolve to nothing the generator was
    given — fabricated provenance. It deliberately says nothing about whether a
    valid citation was the *right* one; that is precision's job.
    """

    valid_count: int = 0
    invalid_count: int = 0
    unknown_labels: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.valid_count + self.invalid_count

    @property
    def validity_rate(self) -> float:
        """Fraction of citations resolving to supplied evidence. No citations -> 1.0."""
        return 1.0 if self.total == 0 else self.valid_count / self.total

    @property
    def hallucinated_citation_rate(self) -> float:
        """Fraction of citations referring to evidence never supplied."""
        return 0.0 if self.total == 0 else self.invalid_count / self.total

    def as_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "valid_citation_count": self.valid_count,
            "invalid_citation_count": self.invalid_count,
            "citation_validity_rate": round(self.validity_rate, 6),
            "hallucinated_citation_rate": round(self.hallucinated_citation_rate, 6),
            "unknown_citation_labels": list(self.unknown_labels),
        }


def citation_validity(
    referenced_labels: Sequence[str],
    unknown_labels: Sequence[str],
) -> CitationValidity:
    """Build a CitationValidity from resolved and unresolved citation labels.

    Takes labels rather than evidence because validity is precisely the question
    of whether a label *could* be resolved; once resolution succeeds the label
    has become evidence and the distinction is gone.
    """
    return CitationValidity(
        valid_count=len(set(referenced_labels)),
        invalid_count=len(set(unknown_labels)),
        unknown_labels=sorted(set(unknown_labels)),
    )


# ---------------------------------------------------------------------------
# Evidence coverage
# ---------------------------------------------------------------------------


def evidence_coverage(
    response_citations: Sequence[CanonicalEvidence],
    target_evidence: Sequence[CanonicalEvidence],
) -> dict[str, bool | None]:
    """Whether the answer's citations cover the expected evidence.

    Text and figure targets are tracked separately, and ``both_targets_covered``
    is None for queries that do not have both kinds — reported as N/A rather
    than counted as a success, matching the PR4 convention.
    """
    cited = _dedupe(response_citations)
    targets = _dedupe(target_evidence)

    text_targets = [t for t in targets if t.figure_id is None]
    figure_targets = [t for t in targets if t.figure_id is not None]

    def covered(subset: Sequence[CanonicalEvidence]) -> bool | None:
        if not subset:
            return None
        return any(t.matches(c) for c in cited for t in subset)

    text_covered = covered(text_targets)
    figure_covered = covered(figure_targets)

    any_covered = bool(targets) and any(_matches_any(c, targets) for c in cited)
    all_covered = bool(targets) and all(any(t.matches(c) for c in cited) for t in targets)

    return {
        "any_target_covered": any_covered if targets else None,
        "all_targets_covered": all_covered if targets else None,
        "text_target_covered": text_covered,
        "figure_target_covered": figure_covered,
        "both_targets_covered": (
            bool(text_covered and figure_covered) if (text_targets and figure_targets) else None
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic grounding proxies (NOT faithfulness)
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    """Content tokens: lowercased, punctuation-stripped, short words dropped.

    Numbers survive the length filter because a wrong figure is exactly the kind
    of unsupported claim worth catching.
    """
    raw = _TOKEN_SPLIT.split((text or "").lower())
    return {
        token
        for token in raw
        if token and (len(token) >= _MIN_SUPPORT_TOKEN_LEN or _NUMERIC.match(token))
    }


def lexical_support_score(answer: str, context: str) -> float:
    """Fraction of the answer's content tokens that also appear in the context.

    A deterministic *proxy*. It underestimates support for correct paraphrase
    and overestimates it for coincidental shared vocabulary, so it is a
    regression signal rather than a semantic verdict.

    Empty answer -> 1.0 (nothing unsupported was asserted).
    Empty context with a non-empty answer -> 0.0 (nothing could support it).
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 1.0
    context_tokens = _tokens(context)
    if not context_tokens:
        return 0.0
    return round(len(answer_tokens & context_tokens) / len(answer_tokens), 6)


def unsupported_claim_fraction(answer: str, context: str, threshold: float = 0.5) -> dict:
    """Sentence-level support proxy.

    A sentence counts as supported when at least ``threshold`` of its content
    tokens appear in the context. The threshold exists because the alternative
    used elsewhere in this repo — "supported if *any* token appears" — marks
    nearly everything supported and is close to useless as a signal.

    Still lexical, and still not a hallucination measurement: a correct
    paraphrase using different vocabulary is scored unsupported.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer or "") if s.strip()]
    if not sentences:
        return {
            "supported_claim_fraction": 1.0,
            "unsupported_claim_fraction": 0.0,
            "claim_count": 0,
            "unsupported_claims": [],
        }

    context_tokens = _tokens(context)
    supported = 0
    unsupported: list[str] = []

    for sentence in sentences:
        sentence_tokens = _tokens(sentence)
        if not sentence_tokens:
            supported += 1  # no content to contradict
            continue
        overlap = len(sentence_tokens & context_tokens) / len(sentence_tokens)
        if overlap >= threshold:
            supported += 1
        else:
            unsupported.append(sentence)

    fraction = supported / len(sentences)
    return {
        "supported_claim_fraction": round(fraction, 6),
        "unsupported_claim_fraction": round(1.0 - fraction, 6),
        "claim_count": len(sentences),
        "unsupported_claims": unsupported[:5],  # capped: artifacts stay readable
    }


def unsupported_numeric_tokens(answer: str, context: str) -> list[str]:
    """Numbers in the answer absent from the context.

    Narrow but high-precision: a fabricated number is unambiguous in a way that
    a paraphrased sentence is not.
    """
    answer_numbers = {t for t in _tokens(answer) if _NUMERIC.match(t)}
    context_numbers = {t for t in _tokens(context) if _NUMERIC.match(t)}
    return sorted(answer_numbers - context_numbers)


# ---------------------------------------------------------------------------
# Size metrics
# ---------------------------------------------------------------------------

_FALLBACK_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> tuple[int, str]:
    """Token count plus the method used, so approximations are never passed off as exact.

    Uses tiktoken's cl100k_base — the encoding the repo's token chunker already
    relies on. It is not the Ollama generator's own tokenizer, so counts are
    comparable across configurations but are not the model's exact accounting.
    """
    if not text:
        return 0, "empty"
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text)), "tiktoken:cl100k_base"
    except Exception:
        return (
            math.ceil(len(text) / _FALLBACK_CHARS_PER_TOKEN),
            f"approx:chars/{_FALLBACK_CHARS_PER_TOKEN}",
        )


def size_metrics(context: str, answer: str) -> dict:
    """Context and answer size, with the counting method recorded."""
    context_tokens, method = count_tokens(context)
    answer_tokens, _ = count_tokens(answer)
    return {
        "context_char_count": len(context or ""),
        "context_token_count": context_tokens,
        "answer_char_count": len(answer or ""),
        "answer_token_count": answer_tokens,
        "token_count_method": method,
    }


__all__ = [
    "CitationValidity",
    "citation_precision_recall",
    "citation_validity",
    "count_tokens",
    "evidence_coverage",
    "lexical_support_score",
    "size_metrics",
    "unsupported_claim_fraction",
    "unsupported_numeric_tokens",
]
