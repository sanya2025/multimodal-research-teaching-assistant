"""mrta.eval — versioned evaluation harness for retrieval benchmarking.

Separate from mrta.evaluation (deepeval-based text eval pipeline).
This module provides:
  - CanonicalEvidence / RetrievedCandidate types (types.py)
  - Retrieval metrics: Recall@k, Hit@k, MRR, nDCG@k (retrieval_metrics.py)
  - Evaluation adapter: maps production Chunk/EvidenceRecord → CanonicalEvidence (adapter.py)
"""
