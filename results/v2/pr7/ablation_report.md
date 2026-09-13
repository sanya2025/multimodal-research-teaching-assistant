# PR7 — Ablation Report

Benchmark **v2** (v2.0.0), 100 queries x 10 configurations. Generated 2026-09-12T04:54:41.449426+00:00.

`candidate_depth=20` · `rrf_k=60` · `final_top_k=5` · commit `9def9b8c85`

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
| `text_only` | 0.2350 | 0.2950 | 0.3292 | 0.2829 | 0.0000 | 0.0000 |
| `caption_only` | 0.1700 | 0.3300 | 0.2835 | 0.2644 | 0.2933 | 0.5467 |
| `clip_only` | 0.1050 | 0.2900 | 0.1983 | 0.2047 | 0.1600 | 0.4800 |
| `text_caption_rrf` | 0.1700 | 0.5150 | 0.3695 | 0.3894 | 0.2933 | 0.4400 |
| `text_clip_rrf` | 0.1050 | 0.4700 | 0.2960 | 0.3262 | 0.1600 | 0.3600 |
| `caption_clip_rrf` | 0.1450 | 0.3400 | 0.2658 | 0.2561 | 0.2533 | 0.5467 |
| `text_caption_clip_rrf` | 0.1450 | 0.3400 | 0.2658 | 0.2561 | 0.2533 | 0.5467 |
| `text_caption_rrf_reranked` | 0.3100 | 0.5250 | 0.4703 | 0.4565 | 0.2133 | 0.3733 |
| `full_reranked` | 0.3500 | 0.6050 | 0.5153 | 0.5111 | 0.2533 | 0.4800 |
| `oracle_evidence_generation` | 0.8750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Generation comparison

| Configuration | Citation P | Citation R | Citation F1 | Validity | Support proxy | Context tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `text_caption_rrf` | 0.6395 | 0.2500 | 0.2073 | 0.9950 | 0.4853 | 625.1100 |
| `text_caption_clip_rrf` | 0.8050 | 0.0400 | 0.0417 | 0.9463 | 0.2654 | 377.2900 |
| `text_caption_rrf_reranked` | 0.7173 | 0.3200 | 0.2685 | 0.9933 | 0.5781 | 803.9500 |
| `full_reranked` | 0.5917 | 0.3100 | 0.2153 | 0.9967 | 0.5618 | 693.7600 |
| `oracle_evidence_generation` | 1.0000 | 0.6050 | 0.6067 | 0.9550 | 0.4356 | 688.7100 |

## Latency (mean ms)

| Configuration | Retrieval | Fusion | Rerank | Generation | Total |
| --- | --- | --- | --- | --- | --- |
| `text_only` | 27.5449 | 0.0000 | 0.0000 | N/A | 27.5449 |
| `caption_only` | 16.8942 | 0.0000 | 0.0000 | N/A | 16.8942 |
| `clip_only` | 16.6224 | 0.0000 | 0.0000 | N/A | 16.6224 |
| `text_caption_rrf` | 44.4392 | 0.3661 | 0.0000 | 1221.0620 | 1265.8672 |
| `text_clip_rrf` | 44.1673 | 0.3188 | 0.0000 | N/A | 44.4861 |
| `caption_clip_rrf` | 33.5167 | 0.1767 | 0.0000 | N/A | 33.6934 |
| `text_caption_clip_rrf` | 61.0615 | 0.2714 | 0.0000 | 791.6254 | 852.9583 |
| `text_caption_rrf_reranked` | 44.4392 | 0.3106 | 77.9418 | 1346.9049 | 1469.5964 |
| `full_reranked` | 61.0615 | 0.3733 | 53.5037 | 1150.1206 | 1265.0590 |
| `oracle_evidence_generation` | 0.0000 | 0.0000 | 0.0000 | 1184.3896 | 1184.3896 |

## By intent

### hard_visual (n=10)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `text_only` | 0.0000 | 0.0000 | 0.0000 |
| `caption_only` | 0.4000 | 0.2583 | 0.4000 |
| `clip_only` | 0.3000 | 0.1700 | 0.3000 |
| `text_caption_rrf` | 0.3000 | 0.2200 | 0.3000 |
| `text_clip_rrf` | 0.2000 | 0.1333 | 0.2000 |
| `caption_clip_rrf` | 0.4000 | 0.2700 | 0.4000 |
| `text_caption_clip_rrf` | 0.4000 | 0.2700 | 0.4000 |
| `text_caption_rrf_reranked` | 0.3000 | 0.1833 | 0.3000 |
| `full_reranked` | 0.5000 | 0.2400 | 0.5000 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

### hybrid (n=25)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `text_only` | 0.3400 | 0.5867 | 0.0000 |
| `caption_only` | 0.3200 | 0.4813 | 0.6400 |
| `clip_only` | 0.2800 | 0.2747 | 0.5600 |
| `text_caption_rrf` | 0.5800 | 0.5780 | 0.5200 |
| `text_clip_rrf` | 0.5200 | 0.4093 | 0.4400 |
| `caption_clip_rrf` | 0.2800 | 0.4380 | 0.5600 |
| `text_caption_clip_rrf` | 0.2800 | 0.4380 | 0.5600 |
| `text_caption_rrf_reranked` | 0.5800 | 0.7200 | 0.3600 |
| `full_reranked` | 0.5400 | 0.6933 | 0.4000 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

### text (n=25)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `text_only` | 0.8400 | 0.7300 | N/A |
| `caption_only` | 0.0000 | 0.0000 | N/A |
| `clip_only` | 0.0000 | 0.0000 | N/A |
| `text_caption_rrf` | 0.6800 | 0.3400 | N/A |
| `text_clip_rrf` | 0.7200 | 0.3480 | N/A |
| `caption_clip_rrf` | 0.0000 | 0.0000 | N/A |
| `text_caption_clip_rrf` | 0.0000 | 0.0000 | N/A |
| `text_caption_rrf_reranked` | 0.7600 | 0.5867 | N/A |
| `full_reranked` | 0.8400 | 0.6713 | N/A |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | N/A |

### visual_caption (n=20)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `text_only` | 0.0000 | 0.0000 | 0.0000 |
| `caption_only` | 0.5000 | 0.3833 | 0.5000 |
| `clip_only` | 0.5000 | 0.2783 | 0.5000 |
| `text_caption_rrf` | 0.5000 | 0.3533 | 0.5000 |
| `text_clip_rrf` | 0.3000 | 0.2200 | 0.3000 |
| `caption_clip_rrf` | 0.6000 | 0.3375 | 0.6000 |
| `text_caption_clip_rrf` | 0.6000 | 0.3375 | 0.6000 |
| `text_caption_rrf_reranked` | 0.4000 | 0.3167 | 0.4000 |
| `full_reranked` | 0.4500 | 0.3267 | 0.4500 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

### visual_layout (n=20)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `text_only` | 0.0000 | 0.0000 | 0.0000 |
| `caption_only` | 0.5500 | 0.3033 | 0.5500 |
| `clip_only` | 0.4500 | 0.2850 | 0.4500 |
| `text_caption_rrf` | 0.3500 | 0.2367 | 0.3500 |
| `text_clip_rrf` | 0.4000 | 0.2467 | 0.4000 |
| `caption_clip_rrf` | 0.5500 | 0.3092 | 0.5500 |
| `text_caption_clip_rrf` | 0.5500 | 0.3092 | 0.5500 |
| `text_caption_rrf_reranked` | 0.4000 | 0.3100 | 0.4000 |
| `full_reranked` | 0.6000 | 0.4242 | 0.6000 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

## By target figure provenance

### nearby_text_fallback (n=38)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `text_only` | 0.1053 | 0.2105 | 0.0000 |
| `caption_only` | 0.4474 | 0.3461 | 0.5263 |
| `clip_only` | 0.4079 | 0.2899 | 0.5000 |
| `text_caption_rrf` | 0.4342 | 0.3535 | 0.3947 |
| `text_clip_rrf` | 0.4211 | 0.3127 | 0.3947 |
| `caption_clip_rrf` | 0.4211 | 0.3009 | 0.5000 |
| `text_caption_clip_rrf` | 0.4211 | 0.3009 | 0.5000 |
| `text_caption_rrf_reranked` | 0.3158 | 0.3465 | 0.2105 |
| `full_reranked` | 0.3421 | 0.3386 | 0.2632 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

### vlm_caption (n=37)

| Configuration | R@5 | MRR@5 | FigR@5 |
| --- | --- | --- | --- |
| `text_only` | 0.1216 | 0.1802 | 0.0000 |
| `caption_only` | 0.4324 | 0.4108 | 0.5676 |
| `clip_only` | 0.3649 | 0.2383 | 0.4595 |
| `text_caption_rrf` | 0.4865 | 0.4059 | 0.4865 |
| `text_clip_rrf` | 0.3514 | 0.2437 | 0.3243 |
| `caption_clip_rrf` | 0.4865 | 0.4095 | 0.5946 |
| `text_caption_clip_rrf` | 0.4865 | 0.4095 | 0.5946 |
| `text_caption_rrf_reranked` | 0.5811 | 0.5189 | 0.5405 |
| `full_reranked` | 0.7162 | 0.5914 | 0.7027 |
| `oracle_evidence_generation` | 1.0000 | 1.0000 | 1.0000 |

## Failure decomposition

| Configuration | Successful | Failed | Failure categories |
| --- | --- | --- | --- |
| `text_only` | 100 | 0 | ranking_miss=3, retrieval_miss=76 |
| `caption_only` | 100 | 0 | ranking_miss=25, retrieval_miss=34 |
| `clip_only` | 100 | 0 | ranking_miss=38, retrieval_miss=26 |
| `text_caption_rrf` | 100 | 0 | answer_support=3, citation_missing=29, citation_relevance=13, ranking_miss=26, retrieval_miss=24 |
| `text_clip_rrf` | 100 | 0 | ranking_miss=35, retrieval_miss=20 |
| `caption_clip_rrf` | 100 | 0 | ranking_miss=33, retrieval_miss=26 |
| `text_caption_clip_rrf` | 100 | 0 | answer_support=3, citation_missing=34, citation_validity=4, ranking_miss=52, retrieval_miss=7 |
| `text_caption_rrf_reranked` | 100 | 0 | answer_support=2, citation_missing=20, citation_relevance=13, citation_validity=1, ranking_miss=29, retrieval_miss=24 |
| `full_reranked` | 100 | 0 | answer_support=1, citation_missing=28, citation_relevance=23, ranking_miss=36, retrieval_miss=7 |
| `oracle_evidence_generation` | 100 | 0 | answer_support=30, citation_missing=40, citation_validity=8 |

