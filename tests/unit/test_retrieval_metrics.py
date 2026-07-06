"""Sanity fixtures for Recall@K and mAP."""
import numpy as np

from mmp_training.retrieval_eval.evaluate import mean_average_precision, rank_of_correct, recall_at_k


def test_perfect_retrieval():
    q = np.eye(3)
    ranks = rank_of_correct(q, q, np.array([0, 1, 2]))
    assert recall_at_k(ranks, 1) == 1.0
    assert mean_average_precision(ranks) == 1.0


def test_worst_case_rank():
    q = np.eye(2)
    ranks = rank_of_correct(q, q, np.array([1, 0]))  # correct item is the least similar
    assert recall_at_k(ranks, 1) == 0.0
    assert recall_at_k(ranks, 2) == 1.0
