"""Retrieval evaluation: Recall@K and mAP on COCO 5k text->image.

Compares measured numbers against published SigLIP-2 results; landing within
tolerance validates the whole embedding+index pipeline before you trust it.
"""
from __future__ import annotations

import numpy as np


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Fraction of queries whose correct image ranks in the top k (ranks are 0-based)."""
    return float((ranks < k).mean())


def mean_average_precision(ranks: np.ndarray) -> float:
    """mAP for the single-relevant-item case: mean of 1/(rank+1)."""
    return float((1.0 / (ranks + 1)).mean())


def rank_of_correct(query_vecs: np.ndarray, image_vecs: np.ndarray, correct_idx: np.ndarray) -> np.ndarray:
    """For each query, the rank of its ground-truth image under cosine similarity.

    Inputs must be L2-normalized. Returns 0-based ranks, shape (num_queries,).
    """
    sims = query_vecs @ image_vecs.T
    order = np.argsort(-sims, axis=1)
    ranks = np.empty(len(query_vecs), dtype=np.int64)
    for i, correct in enumerate(correct_idx):
        ranks[i] = int(np.where(order[i] == correct)[0][0])
    return ranks
