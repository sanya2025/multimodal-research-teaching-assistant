"""mrta.eval.generation_conditions — frozen PR8 generation strategies.

PR8 asks one question: given *identical* retrieved evidence, does presenting it
differently or asking for structured output change how completely the generator
cites it?

    frozen top-5 evidence ──┬─► G0 baseline prompt          ─► prose
                            ├─► G1 explicit citation rules  ─► prose
                            ├─► G2 structured evidence cards─► prose
                            └─► G3 cards + JSON contract    ─► {answer, citations}

The conditions differ only in prompt and output parsing. Retrieval, fusion,
reranking and the evidence itself are untouched, and ``evidence_context_hash``
exists to prove it rather than assert it.

The model selects labels; the application owns canonical identity. A label the
model invents resolves to nothing and is recorded as invalid — it never becomes
a document id, page or figure id in a result.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# Condition identifiers. Stable, because results are keyed by them.
G0_BASELINE = "g0_baseline"
G1_EXPLICIT_CITATIONS = "g1_explicit_citations"
G2_STRUCTURED_EVIDENCE = "g2_structured_evidence"
G3_STRUCTURED_GENERATION = "g3_structured_generation"

# Failure taxonomy for the generation stage (spec section 36). Retrieval and
# ranking failures stay in the PR7 taxonomy; these describe only what the
# generator did with evidence it was actually given.
FAILURE_MISSING_EXPECTED_CITATION = "missing_expected_citation"
FAILURE_IRRELEVANT_VALID_CITATION = "irrelevant_valid_citation"
FAILURE_INVALID_CITATION = "invalid_citation"
FAILURE_MALFORMED_STRUCTURED_OUTPUT = "malformed_structured_output"
FAILURE_UNCITED_CLAIM = "uncited_claim"
FAILURE_EMPTY_ANSWER = "empty_answer"
FAILURE_GENERATION_ERROR = "generation_error"

_LABEL_PATTERN = re.compile(r"\[([TF])(\d+)\]")
_BARE_LABEL_PATTERN = re.compile(r"^([TF])(\d+)$")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


@dataclass(frozen=True)
class GenerationCondition:
    """One frozen generation strategy.

    ``template`` selects the prompt; ``structured_output`` decides whether the
    generator is asked for JSON and whether the reply is parsed as such. Nothing
    else varies between conditions — that is what makes the comparison causal.
    """

    condition_id: str
    template: str
    structured_output: bool = False
    description: str = ""

    def prompt_hash(self) -> str:
        """Hash of the rendered template source, recorded for reproducibility.

        Hashing the template rather than a rendered prompt keeps the value
        stable across queries while still changing if anyone edits the wording
        after the conditions were frozen.
        """
        from pathlib import Path

        import mrta.prompts as prompts_pkg

        path = Path(prompts_pkg.__file__).parent / f"{self.template}.j2"
        if not path.exists():
            return "missing"
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# Frozen before the final comparative run (spec sections 7, 23).
FROZEN_CONDITIONS: tuple[GenerationCondition, ...] = (
    GenerationCondition(
        G0_BASELINE,
        template="canonical_multimodal_rag",
        description="PR6/PR7 production prompt, unchanged",
    ),
    GenerationCondition(
        G1_EXPLICIT_CITATIONS,
        template="g1_explicit_citations",
        description="G0 evidence layout, explicit claim-level citation rules",
    ),
    GenerationCondition(
        G2_STRUCTURED_EVIDENCE,
        template="g2_structured_evidence",
        description="G1 rules, evidence presented as typed cards",
    ),
    GenerationCondition(
        G3_STRUCTURED_GENERATION,
        template="g3_structured_generation",
        structured_output=True,
        description="G2 cards plus a JSON {answer, citations} contract",
    ),
)

CONDITIONS_BY_ID: dict[str, GenerationCondition] = {c.condition_id: c for c in FROZEN_CONDITIONS}


@dataclass
class ParsedGeneration:
    """What the application extracted from one generator reply.

    ``labels`` are the label strings the model referenced, before resolution.
    ``malformed`` records that a structured reply could not be parsed — the run
    degrades to prose extraction rather than pretending the parse succeeded.
    """

    answer: str
    labels: list[str] = field(default_factory=list)
    malformed: bool = False
    parse_note: str | None = None


def _normalise_label(raw: str) -> str | None:
    """Accept "T1", "[T1]", " f2 " → "[T1]"/"[F2]"; reject anything else.

    The model is asked for bare labels in G3 and bracketed labels inline, so
    both spellings are legitimate. Anything that is not a label shape at all —
    a page number, a filename, a sentence — returns None and is counted invalid
    rather than coerced into something that looks like provenance.
    """
    token = (raw or "").strip().strip("[]").strip().upper()
    match = _BARE_LABEL_PATTERN.match(token)
    return f"[{match.group(1)}{match.group(2)}]" if match else None


def extract_inline_labels(answer: str) -> list[str]:
    """Labels cited inline in prose, in first-appearance order."""
    seen: list[str] = []
    for kind, number in _LABEL_PATTERN.findall(answer or ""):
        label = f"[{kind}{number}]"
        if label not in seen:
            seen.append(label)
    return seen


def parse_structured_generation(raw: str) -> ParsedGeneration:
    """Parse a G3 reply as ``{"answer": str, "citations": [str]}``.

    Ollama's JSON mode is not a guarantee, so every failure mode degrades to
    prose extraction with ``malformed=True`` recorded. Silently succeeding on a
    bad parse would turn a generation failure into a citation failure and
    misattribute it in the taxonomy.
    """
    text = (raw or "").strip()
    if not text:
        return ParsedGeneration(answer="", labels=[], malformed=True, parse_note="empty reply")

    # Models sometimes wrap JSON in fences despite being told not to.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedGeneration(
            answer=raw,
            labels=extract_inline_labels(raw),
            malformed=True,
            parse_note=f"invalid JSON: {exc.msg}",
        )

    if not isinstance(payload, dict):
        return ParsedGeneration(
            answer=raw,
            labels=extract_inline_labels(raw),
            malformed=True,
            parse_note=f"expected object, got {type(payload).__name__}",
        )

    answer = payload.get("answer")
    if not isinstance(answer, str):
        return ParsedGeneration(
            answer=raw,
            labels=extract_inline_labels(raw),
            malformed=True,
            parse_note="missing or non-string 'answer'",
        )

    raw_citations = payload.get("citations", [])
    if not isinstance(raw_citations, list):
        # The answer is usable even when the citation list is not; fall back to
        # whatever the prose cited rather than discarding a valid answer.
        return ParsedGeneration(
            answer=answer,
            labels=extract_inline_labels(answer),
            malformed=True,
            parse_note="'citations' is not a list",
        )

    labels: list[str] = []
    for item in raw_citations:
        if not isinstance(item, str):
            continue
        normalised = _normalise_label(item)
        if normalised and normalised not in labels:
            labels.append(normalised)

    # A structured reply that cites nothing in its list but cites inline is
    # honouring the contract loosely; take the inline labels rather than
    # recording a false zero.
    if not labels:
        labels = extract_inline_labels(answer)

    return ParsedGeneration(answer=answer, labels=labels, malformed=False)


def parse_generation(raw: str, condition: GenerationCondition) -> ParsedGeneration:
    """Parse a generator reply according to its condition's output contract."""
    if condition.structured_output:
        return parse_structured_generation(raw)
    return ParsedGeneration(answer=raw or "", labels=extract_inline_labels(raw))


def evidence_context_hash(labelled: dict[str, dict]) -> str:
    """Stable hash of the evidence supplied to the generator.

    Covers the label, canonical identity and the evidence text, so any change to
    *what* the generator sees changes the hash. PR8's causal claim depends on
    this being identical across G0-G3 for a given query; the hash is what makes
    that checkable instead of assumed.
    """
    parts: list[str] = []
    for label in sorted(labelled):
        entry = labelled[label]
        evidence = entry["evidence"]
        parts.append(
            "|".join(
                [
                    label,
                    evidence.document_id,
                    str(evidence.page_number),
                    str(evidence.figure_id),
                    hashlib.sha256((entry.get("text") or "").encode()).hexdigest()[:16],
                ]
            )
        )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def claim_citation_coverage(answer: str) -> dict:
    """Fraction of sentences carrying at least one citation label.

    Measures *citation behaviour only*. A sentence may be perfectly grounded and
    uncited, or cited and wrong; this counts neither. It is not a faithfulness
    or correctness signal and must not be reported as one.

    Sentences with no alphabetic content (stray fragments, list bullets) are
    skipped rather than counted as uncited claims.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer or "") if s.strip()]
    substantive = [s for s in sentences if any(ch.isalpha() for ch in s)]
    if not substantive:
        return {
            "claim_count": 0,
            "cited_claim_count": 0,
            "claim_citation_coverage": None,
            "uncited_claim_fraction": None,
        }

    cited = sum(1 for s in substantive if _LABEL_PATTERN.search(s))
    coverage = cited / len(substantive)
    return {
        "claim_count": len(substantive),
        "cited_claim_count": cited,
        "claim_citation_coverage": round(coverage, 6),
        "uncited_claim_fraction": round(1.0 - coverage, 6),
    }


def classify_generation_failures(
    *,
    answer: str,
    citation_scores: dict,
    invalid_count: int,
    malformed: bool,
    claim_coverage: float | None,
    has_targets: bool,
    uncited_threshold: float = 0.5,
) -> list[str]:
    """All generation-stage failure modes present for one answer.

    A list rather than a single category: unlike PR7's earliest-stage retrieval
    attribution, these are not mutually exclusive. One answer can both omit an
    expected citation and add an irrelevant one, and collapsing that to a single
    label would lose half the signal.
    """
    failures: list[str] = []

    if not (answer or "").strip():
        failures.append(FAILURE_EMPTY_ANSWER)
    if malformed:
        failures.append(FAILURE_MALFORMED_STRUCTURED_OUTPUT)
    if invalid_count > 0:
        failures.append(FAILURE_INVALID_CITATION)
    if has_targets:
        if citation_scores.get("citation_recall", 1.0) < 1.0:
            failures.append(FAILURE_MISSING_EXPECTED_CITATION)
        if citation_scores.get("citation_precision", 1.0) < 1.0:
            failures.append(FAILURE_IRRELEVANT_VALID_CITATION)
    if claim_coverage is not None and claim_coverage < uncited_threshold:
        failures.append(FAILURE_UNCITED_CLAIM)

    return failures


def paired_comparison(
    baseline: Sequence[float | None],
    treatment: Sequence[float | None],
) -> dict:
    """Win/tie/loss counts for a paired per-query metric.

    Both conditions run on the same queries with the same evidence, so a paired
    comparison is the honest summary: a mean difference can hide that a change
    helps a few queries a lot while hurting many slightly.

    Pairs where either side is None are excluded and counted, so an absent
    metric never reads as a tie.
    """
    wins = ties = losses = skipped = 0
    deltas: list[float] = []

    for base, treat in zip(baseline, treatment):
        if base is None or treat is None:
            skipped += 1
            continue
        delta = treat - base
        deltas.append(delta)
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1

    mean_delta = round(sum(deltas) / len(deltas), 6) if deltas else None
    return {
        "treatment_better": wins,
        "tie": ties,
        "baseline_better": losses,
        "skipped": skipped,
        "compared": len(deltas),
        "mean_delta": mean_delta,
    }


__all__ = [
    "CONDITIONS_BY_ID",
    "FROZEN_CONDITIONS",
    "G0_BASELINE",
    "G1_EXPLICIT_CITATIONS",
    "G2_STRUCTURED_EVIDENCE",
    "G3_STRUCTURED_GENERATION",
    "GenerationCondition",
    "ParsedGeneration",
    "claim_citation_coverage",
    "classify_generation_failures",
    "evidence_context_hash",
    "extract_inline_labels",
    "paired_comparison",
    "parse_generation",
    "parse_structured_generation",
]
