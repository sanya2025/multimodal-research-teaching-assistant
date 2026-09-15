# ADR-012 — Two-Tier CI Quality Gates and Evaluation Regression Protection

**Date:** 2026-09-15
**Status:** Accepted
**Branch:** `ci/two-tier-quality-gates` (PR9)
**Relates to:** [ADR-006](ADR-006-evaluation-framework.md), [ADR-007](ADR-007-cross-encoder-reranking.md), [ADR-010](ADR-010-ablation-and-generation-evaluation.md), [ADR-011](ADR-011-citation-aware-generation.md)

---

## Context

PR1–PR8 produced a measured multimodal system: a frozen 100-query benchmark, a
production-equivalent retrieval configuration, and a reusable ablation runner.
Nothing protected any of it. CI ran lint, types, the unit suite and a Docker
build — none of which would notice if a refactor moved Recall@5 from 0.605 to
0.48. The numbers in `results/v2/` were assertions about a past commit, not
properties of `main`.

The obvious fix — run the benchmark on every pull request — fails on cost and on
availability. It needs model weights and, as inspection showed, it could not run
on a hosted runner at all.

## Decisions

### 1. Two tiers, split by what they protect rather than by what they cost

**Tier 1 (`MRTA Fast PR CI`)** protects the code: lint, format, types, the fast
test suite, and a synthetic smoke check of the evaluation plumbing. Every pull
request, CPU-only, no weights, no services, no secrets — so it works for forks.

**Tier 2 (`MRTA Evaluation Regression`)** protects the measurement: the frozen v2
retrieval evaluation compared against an immutable baseline. Push to `main`,
nightly at 03:17 UTC, and manual dispatch.

The split is not "fast things and slow things". It is that fast review feedback
and a trustworthy measurement have different budgets, and merging them makes
each worse: a benchmark on every PR is unaffordable, and a benchmark on no
branch is unprotected.

"Gate" means two different things across the tiers, and conflating them is the
mistake this ADR most wants to prevent:

| | Tier 1 | Tier 2 |
|---|---|---|
| Runs on | pull requests | `main`, nightly, dispatch |
| Kind | **pre-merge quality gate** | **post-merge regression detector** |
| Can block a merge | yes, if configured as required | no |
| Failure means | do not merge this yet | a regression already reached `main` |

Tier 2 can fail loudly and make a regression unmissable, but it cannot prevent
the commit that caused it from landing. That is a deliberate consequence of not
running the benchmark on pull requests, not a gap to be patched. Making the
0.6050 / 0.5153 / 0.4800 contract merge-blocking would require running Tier 2 on
`pull_request` or building a selective pre-merge evaluation — either of which
reintroduces the cost this split exists to avoid.

### 2. The frozen benchmark is now reproducible without a model server

This was a blocker, not a preference, and inspection is what surfaced it.

The v2 text and caption indices were built with `nomic-embed-text`, which
`Embedder` serves over the Ollama REST API. That dependency is not build-time
only: `VectorStore.search` embeds the *query* through the same embedder, so
Ollama was required at each of the 100 evaluation queries. Meanwhile
`.gitignore` excluded `*.faiss` and `data/vector_store/`, so `data/vector_store/v2_corpus`
had **zero** tracked files. A hosted runner could neither load the indices nor
rebuild them.

Three things change:

- The 2.8 MB of v2 index artifacts are un-ignored and committed. They are small,
  they are the measurement's substrate, and the blanket "vector stores are large
  and rebuildable" rule was true of everything except these.
- `data/eval/query_embeddings_v2.npz` (284 KB) freezes the 100 query vectors,
  produced once by the real embedder via `scripts/build_query_embeddings.py`.
- `run_ablation.py --offline` reads them through `CachedQueryEmbedder` instead of
  calling a service.

Retrieval is unchanged — the same vectors reach the same indices in the same
order. The offline run reproduces PR5/PR7 **bit-exactly** on all nine retrieval
metrics, which is the evidence that this is a packaging change and not a
scientific one.

Crucially, the offline branch never reads `settings.embedding_model`. It cannot:
the model name comes from the cache, which was built from the index's own
`config.json`. This matters because the settings chain resolves to
`all-MiniLM-L6-v2` in two of the three tracked paths — the `Settings` default and
`configs/test.yaml` — and Tier 2 runs with `MRTA_ENV=test`. Verified by forcing
`EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` as an environment
variable, which outranks every other source: the offline run still recorded
`nomic-embed-text` and reproduced all nine metrics exactly. Under the same
forcing, the live path fails at the first query on FAISS's dimension assertion
rather than producing numbers.

Note that neither `VectorStore.load` nor `CaptionVectorStore.load` validates the
supplied embedder against the `config.json` it wrote — both only document the
precondition. That is left as-is: it is production code outside PR9's scope, and
the offline path removes the choice rather than checking it after the fact.

A side effect worth stating plainly: before PR9, the frozen results depended on
`EMBEDDING_MODEL=nomic-embed-text` in an **untracked `.env`**, while tracked
config said `all-MiniLM-L6-v2`. A fresh clone reproduced nothing. It does now.

The alternative — installing Ollama on the runner — was rejected: it preserves
the numbers equally well but leaves the clean-clone reproducibility gap open,
and makes every Tier-2 run depend on a third-party install script.

### 3. The cache fails closed, and has no fallback

`CachedQueryEmbedder` raises on any query it was not built for, and
`verify_queries` compares a SHA-256 of the benchmark file before retrieval
starts. A live-embedder fallback was deliberately not added: a silent miss would
mix vectors from two models and report the difference as a retrieval regression.
Loud failure on a drifted benchmark is the cheaper error.

### 4. The baseline is `full_reranked`, sourced from PR7, cross-checked against PR5

PR5's `rerank_text_caption_clip` and PR7's `full_reranked` are the same pipeline
— Text + Caption + CLIP → equal-weight RRF → top-20 → CrossEncoder → top-5 — and
agree on every retrieval metric. PR7 is the source because its artifact records
the embedding model, candidate depth, `rrf_k` and `final_top_k` in
machine-readable metadata; PR5's does not name the embedding model, which is the
one field that turned out to matter most.

`results/v2/baselines/retrieval_regression_v1.json` is a dedicated immutable
artifact rather than a pointer at a PR's output, so a future re-run of PR7 cannot
silently move the baseline. Tests assert it still matches both source artifacts.

### 5. Tolerances are absolute metric points, not percentages or statistics

```text
Recall@5         max allowed absolute drop  0.02
MRR@5            max allowed absolute drop  0.02
Figure Recall@5  max allowed absolute drop  0.03
```

A baseline of 0.605 against a current of 0.580 is a drop of 0.025 and fails.

Figure Recall@5 gets more room because it is computed over the 21 canonical
figures rather than all 100 queries: one figure moving is worth ~0.013, an order
of magnitude coarser than one query moving Recall@5. A 0.02 gate there would fire
on a single figure changing rank.

These are engineering tolerances chosen to sit above run-to-run noise. They are
not confidence intervals, and a pass is not evidence that nothing changed. The
benchmark is deterministic given fixed indices and a fixed seed, so the expected
drift is zero; the tolerance exists for library-version and platform effects, not
for sampling.

### 6. No citation metric is a CI gate

PR8 measured that structured generation raises citation recall (0.3100 → 0.4300)
while lowering precision (0.5883 → 0.3500), and could not say whether the answers
got better — ADR-011 §8 and its Consequences. Gating on either side of that
trade-off would freeze one answer to an open question into CI, and would cause
future work that improves answers to fail the build.

So Tier 2 gates retrieval only. Citation precision, citation recall, citation F1,
lexical support and claim-citation coverage are measured and reported by the
ablation runner; none of them blocks a merge. Nothing here introduces a semantic
judge, and PR8's G0–G3 conditions are untouched.

### 7. Every failure mode fails closed

Missing baseline, missing current file, malformed JSON, an absent metric, a
non-numeric or NaN/Infinity value, or a configuration mismatch all exit non-zero.
The gate never treats "could not compare" as "nothing regressed" — that would
make it decoration that looks like evidence.

Configuration identity is checked before any verdict is trusted: benchmark name
and dataset version, configuration id, candidate depth, `rrf_k`, final top-k, and
**all three retrieval models** — embedding, CLIP and reranker. A `full_reranked`
baseline against a `text_caption_rrf` current run fails **even when every metric
is better**, because the comparison is meaningless rather than favourable.

All three models are checked, not just the reranker, because dimension is not a
safety net. A 384-d MiniLM query against the 768-d v2 index trips FAISS's
`assert d == self.d`; a *different 768-d* embedder, or a different 512-d CLIP,
changes every score and raises nothing. Given that the embedding model is the
exact field this PR found mis-specified (decision 2), leaving it out of the
identity check would have been the one omission most likely to matter.

### 8. Markers are applied where they are true, not by directory

`unit`, `integration` and `eval` follow the directory layout and are applied
automatically in `tests/conftest.py` — repeating them across 37 files would be 37
chances to drift. `heavy` is explicit, because loading real weights is a property
of a test, not of a location: two heavyweight groups live in `tests/unit`, which
is otherwise entirely fast.

Tier 1 selects `-m "not heavy"` rather than the more obvious
`-m "not heavy and not eval"`. Now that the benchmark assets are committed, the
`eval` tests read tracked JSON in milliseconds and are exactly the checks that
catch a corrupted benchmark before it reaches `main`. Excluding them would trade
real coverage for nothing.

This also fixed an unnoticed cost: `tests/unit/test_clip_embedder.py` and
`test_vector_store.py::TestEmbedder` construct real models, so CI had been
downloading ~700 MB of weights on every run, invisible behind a 12-second suite.
Tier 1 now sets `HF_HUB_OFFLINE=1`, which turns a future mistake of that kind
into a loud failure rather than a slow one.

### 9. Existing coverage moves, it does not disappear

| Pre-PR9 check | Now runs in |
|---|---|
| `ruff check` | Tier 1 — `lint-and-type` |
| `black --check` | Tier 1 — `lint-and-type` |
| `mypy` | Tier 1 — `lint-and-type` |
| `pytest` (fast) | Tier 1 — `fast-tests` |
| `pytest` (real weights) | Tier 2 — `heavy-tests` |
| `pip-audit` | `CI / audit`, unchanged |
| Docker build + `/health` | `CI / docker`, unchanged |

`CI / audit` and `CI / docker` keep their job names and pull-request trigger so
existing branch protection still resolves; they no longer chain behind `needs: test`,
which used to serialise the whole workflow. **`CI / test` and `CI / type-check`
no longer exist**; their replacements are `MRTA Fast PR CI / fast-tests` and
`MRTA Fast PR CI / lint-and-type`.

Checked at the time of writing: `main` has no classic branch protection and no
rulesets, so no required status check referenced the removed names and nothing
needed repointing. If protection is added later, see decision 10.

### 10. Tier 1 may be required; Tier 2 must never be

> Tier 1 checks may be configured as required pre-merge status checks. Tier 2
> intentionally does not run on pull requests and therefore **must not** be
> configured as a required PR status check; it detects evaluation regressions on
> `main` rather than preventing the triggering merge.

A required check that can never report leaves every pull request permanently
pending, and the repository is then unmergeable until someone with admin rights
works out why. Recording the rule here is cheaper than rediscovering it.

The eligible set, by the names GitHub registers (`workflow name / job name`):

| Check | Required-eligible |
|---|---|
| `MRTA Fast PR CI / lint-and-type` | yes |
| `MRTA Fast PR CI / fast-tests` | yes |
| `MRTA Fast PR CI / smoke-eval` | yes |
| `CI / docker` | yes |
| `CI / audit` | no — `continue-on-error: true`, so it cannot fail |
| `MRTA Evaluation Regression / full-evaluation-gate` | **no — never runs on PRs** |
| `MRTA Evaluation Regression / heavy-tests` | **no — never runs on PRs** |

Order of operations matters: GitHub only offers a context it has already seen, so
protection is configured *after* the first hosted run, never before. Enabling it
first leaves the named contexts pending forever — the same failure as requiring a
Tier-2 job, arrived at from the other direction.

`enforce_admins: false` is the right starting point on a single-maintainer
research repository: it keeps an escape hatch while the architecture settles, and
can be tightened once the tiers have run over several pull requests.

## Consequences

**Positive**

- A retrieval regression beyond tolerance now fails a build instead of being
  discovered whenever someone next re-ran an evaluation by hand.
- The frozen v2 benchmark reproduces from a clean clone for the first time; the
  measurement no longer depends on one machine's untracked `.env`.
- Tier 1 jobs run in parallel rather than behind `needs: test`.
- The offline path also makes local evaluation faster and dependency-free: the
  full `full_reranked` run takes **5.3 s** with no model server.

**Negative / tradeoffs**

- 3.1 MB of binary artifacts are now versioned. They are regenerable but not
  diffable, and a future embedding-model change invalidates all of them together.
- The query cache must be rebuilt by hand if the benchmark queries change. That
  is guarded by a hash check that fails closed, so the cost is a clear error
  rather than a wrong number.
- Model revisions are pinned by name, not by hub revision hash. A silent upstream
  re-upload of `cross-encoder/ms-marco-MiniLM-L-6-v2` would change results without
  changing any identifier we record. Documented rather than fixed: pinning
  revisions is a real change to how models are loaded and belongs in its own PR.
- Tier 2 gates one configuration on one benchmark. It says nothing about the
  other nine ablation configurations, or about generalization beyond v2.
- Tier 1 still installs torch, because `sentence-transformers` requires it. The
  dominant Tier-1 cost is dependency installation, not tests.

> Passing Tier 2 means that selected frozen retrieval metrics have not regressed
> beyond configured tolerances on the v2 benchmark. It does not prove semantic
> answer correctness, generation faithfulness, or generalization beyond the
> benchmark.

## Alternatives considered

**Run the benchmark on every PR.** Rejected on cost, and impossible before the
offline path existed.

**Install Ollama on the Tier-2 runner.** Rejected — see decision 2. It preserves
the numbers but leaves clean-clone reproducibility broken.

**Rebuild the indices in CI from the committed PDFs.** Rejected: it needs the
same embedding service, and it would make every run's numbers depend on index
construction being bit-reproducible, which has never been tested.

**Git LFS for the artifacts.** Unnecessary at 3.1 MB, and it would add a
checkout dependency to every clone and CI job.

**Statistical significance testing instead of tolerances.** Rejected as a
category error: the benchmark is deterministic given fixed indices, so there is
no sampling distribution to test. Absolute tolerances say what they mean.

**Lower the baseline if Tier 2 fails.** Explicitly forbidden. A deliberate
baseline change requires a new `baseline_version` and written justification.

## Related ADRs

- [ADR-006 — Evaluation Framework](ADR-006-evaluation-framework.md)
- [ADR-010 — Ablation Framework and Generation Evaluation](ADR-010-ablation-and-generation-evaluation.md)
- [ADR-011 — Citation-Aware Generation](ADR-011-citation-aware-generation.md)
