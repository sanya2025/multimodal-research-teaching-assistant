# PR7 — Ablation Report

Benchmark **v2** (v2.0.0), 100 queries x 2 configurations. Generated 2026-09-14T06:01:11.792967+00:00.

`candidate_depth=20` · `rrf_k=60` · `final_top_k=5` · commit `bb9a81942c`

Models: embedding_model=nomic-embed-text, clip_model=openai/clip-vit-base-patch32, reranker_model=cross-encoder/ms-marco-MiniLM-L-6-v2, generator_model=llama3.2:latest

> **Support metrics are deterministic lexical proxies.** They are useful for
> regression testing, but they underestimate support for correct paraphrases
> and must not be read as a complete semantic faithfulness metric.

> **`oracle_evidence_generation` is evaluation-only.** It receives ground-truth
> evidence directly, so its retrieval metrics are perfect by construction and
> are never comparable with a retrieval configuration.

## Retrieval comparison

| Configuration | R@1 | R@5 | MRR@5 | nDCG@5 | FigR@1 | FigR@5 |
| --- | --- | --- | --- | --- | --- | --- |
| `full_reranked` | 0.3500 | 0.6050 | 0.5153 | 0.5111 | 0.2533 | 0.4800 |
| `oracle_evidence_generation` | 0.8750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Generation comparison

| Configuration | Citation P | Citation R | Citation F1 | Validity | Support proxy | Context tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `full_reranked` | 0.5151 | 0.3362 | 0.2653 | 0.9854 | 0.6254 | 693.7600 |
| `oracle_evidence_generation` | 1.0000 | 0.6200 | 0.6267 | 0.9241 | 0.4704 | 688.7100 |

## Latency (mean ms)

| Configuration | Retrieval | Fusion | Rerank | Generation | Total |
| --- | --- | --- | --- | --- | --- |
| `full_reranked` | 56.7631 | 0.3577 | 56.6891 | 1229.2116 | 1343.1599 |
| `oracle_evidence_generation` | 0.0000 | 0.0000 | 0.0000 | 1099.7317 | 1099.9495 |

## By intent

### hard_visual (n=40)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `full_reranked` | 0.5000 | 0.2400 | 0.5000 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

### hybrid (n=100)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `full_reranked` | 0.5400 | 0.6933 | 0.4000 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

### text (n=100)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `full_reranked` | 0.8400 | 0.6713 | N/A |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | N/A |

### visual_caption (n=80)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `full_reranked` | 0.4500 | 0.3267 | 0.4500 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

### visual_layout (n=80)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `full_reranked` | 0.6000 | 0.4242 | 0.6000 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

## By target figure provenance

### nearby_text_fallback (n=152)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `full_reranked` | 0.3421 | 0.3386 | 0.2632 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

### vlm_caption (n=148)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `full_reranked` | 0.7162 | 0.5914 | 0.7027 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

## PR8 generation conditions

Comparison configuration: **`full_reranked`**. These numbers are for that configuration alone — the oracle is reported separately below and is never averaged in, because it is handed ground-truth evidence and scores near perfectly by construction.

Evidence-hash integrity: **200/200** (query, configuration) pairs supplied identical evidence to every condition (PASS). Retrieval is frozen; only the prompt and output contract vary.

> **G3 is a combined intervention.** It changes two things relative to G2 — typed evidence cards *and* the JSON `{answer, citations}` contract — so its improvement demonstrates the effect of the combination, **not** of JSON output alone. Separating them would need a 'G0 prompt + JSON output' cell, which is absent from the frozen matrix; adding one after seeing results would be tuning rather than measurement.

| Condition | Cit P | Cit R | Cit F1 | Validity | Fig coverage | Hybrid both | Claim citation cov |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `g0_baseline` | 0.5883 | 0.3100 | 0.2203 | 0.9967 | 0.2267 | 0.2000 | 0.2650 |
| `g1_explicit_citations` | 0.5770 | 0.3200 | 0.2593 | 0.9817 | 0.2400 | 0.1200 | 0.3306 |
| `g2_structured_evidence` | 0.5450 | 0.2850 | 0.2213 | 0.9733 | 0.2133 | 0.1200 | 0.3927 |
| `g3_structured_generation` | 0.3500 | 0.4300 | 0.3603 | 0.9900 | 0.3467 | 0.2000 | 0.6015 |

### Context cost

| Condition | Context tokens | Answer tokens | Generation mean ms |
| --- | --- | --- | --- |
| `g0_baseline` | 693.7600 | 122.8400 | 1409.0240 |
| `g1_explicit_citations` | 693.7600 | 114.5600 | 1338.9846 |
| `g2_structured_evidence` | 693.7600 | 120.8500 | 1495.2446 |
| `g3_structured_generation` | 693.7600 | 53.7200 | 673.5932 |

### Paired comparison (per-query, same evidence)

| Comparison | Metric | Treatment better | Tie | Baseline better | Mean Δ |
| --- | --- | --- | --- | --- | --- |
| `g1_explicit_citations_vs_g0_baseline` | citation_recall | 11 | 80 | 9 | 0.0100 |
| `g1_explicit_citations_vs_g0_baseline` | citation_f1 | 15 | 72 | 13 | 0.0390 |
| `g1_explicit_citations_vs_g0_baseline` | citation_precision | 17 | 63 | 20 | -0.0113 |
| `g1_explicit_citations_vs_g0_baseline` | claim_citation_coverage | 42 | 35 | 23 | 0.0656 |
| `g2_structured_evidence_vs_g1_explicit_citations` | citation_recall | 14 | 71 | 15 | -0.0350 |
| `g2_structured_evidence_vs_g1_explicit_citations` | citation_f1 | 18 | 57 | 25 | -0.0380 |
| `g2_structured_evidence_vs_g1_explicit_citations` | citation_precision | 24 | 49 | 27 | -0.0320 |
| `g2_structured_evidence_vs_g1_explicit_citations` | claim_citation_coverage | 47 | 16 | 37 | 0.0621 |
| `g3_structured_generation_vs_g2_structured_evidence` | citation_recall | 20 | 74 | 6 | 0.1450 |
| `g3_structured_generation_vs_g2_structured_evidence` | citation_f1 | 31 | 57 | 12 | 0.1390 |
| `g3_structured_generation_vs_g2_structured_evidence` | citation_precision | 18 | 44 | 38 | -0.1950 |
| `g3_structured_generation_vs_g2_structured_evidence` | claim_citation_coverage | 53 | 22 | 25 | 0.2088 |
| `g3_structured_generation_vs_g0_baseline` | citation_recall | 23 | 67 | 10 | 0.1200 |
| `g3_structured_generation_vs_g0_baseline` | citation_f1 | 35 | 52 | 13 | 0.1400 |
| `g3_structured_generation_vs_g0_baseline` | citation_precision | 20 | 35 | 45 | -0.2383 |
| `g3_structured_generation_vs_g0_baseline` | claim_citation_coverage | 65 | 12 | 23 | 0.3365 |

### Oracle condition (diagnostic only)

The oracle receives the benchmark's ground-truth evidence directly, so its retrieval metrics are perfect by construction. It bounds how completely the generator cites evidence when availability is not the constraint. It is **not** comparable with a retrieval configuration and must never be averaged with one.

| Condition | Cit R | Cit F1 | Claim citation cov |
| --- | --- | --- | --- |
| `g0_baseline` | 0.6050 | 0.6067 | 0.2840 |
| `g1_explicit_citations` | 0.5050 | 0.5133 | 0.2595 |
| `g2_structured_evidence` | 0.4450 | 0.4533 | 0.2386 |
| `g3_structured_generation` | 0.9250 | 0.9333 | 0.4248 |

## Failure decomposition

| Configuration | Successful | Failed | Failure categories |
| --- | --- | --- | --- |
| `full_reranked` | 400 | 0 | answer_support=7, citation_missing=108, citation_relevance=76, citation_validity=3, ranking_miss=144, retrieval_miss=28 |
| `oracle_evidence_generation` | 400 | 0 | answer_support=88, citation_missing=154, citation_validity=49 |

