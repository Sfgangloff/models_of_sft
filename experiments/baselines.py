"""
baselines.py — non-neural reference predictors for T1.

frequency_learner: the Bayes-optimal *behavioral* rule-inducer in the
infinite-context limit. It collects every domino occurring in the K context
configs and predicts the query admissible iff every domino in the query was
observed. Strong finite-K reference; uses only context, so it is independent of
the number of training worlds M (a flat reference line on the headline plot).
"""

from __future__ import annotations
import numpy as np
from worlds import observed_dominoes, config_dominoes


def frequency_learner_predict(context, query, q):
    """context (K,N,N), query (N,N) -> predicted label in {0,1}."""
    seen = observed_dominoes(context, q)
    qd = config_dominoes(query)
    return int(qd <= seen)        # admissible iff all query dominoes were seen


def frequency_learner_batch(context, query, q):
    """context (B,K,N,N), query (B,N,N) -> predictions (B,)."""
    B = query.shape[0]
    return np.array([frequency_learner_predict(context[b], query[b], q)
                     for b in range(B)], dtype=np.int64)
