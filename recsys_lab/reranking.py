"""Diversity-aware reranking on a fixed candidate list."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


def rerank_mmr(candidate_ids, relevance_scores, item_features, k=5, diversity_weight=0.3):
    """Greedily trade normalized relevance against similarity to prior picks.

    Arrays must follow candidate_ids order. Relevance is min-max scaled within
    this candidate list; similarity is cosine similarity clipped below at zero.
    diversity_weight=0 preserves the original relevance ordering on score ties.
    """

    if k <= 0 or not 0 <= diversity_weight <= 1:
        raise ValueError("need positive k and diversity_weight between 0 and 1")
    ids = list(candidate_ids)
    scores = np.asarray(relevance_scores, dtype=float)
    features = np.asarray(item_features, dtype=float)
    if len(ids) != len(set(ids)) or scores.shape != (len(ids),):
        raise ValueError("need unique IDs and one score per candidate")
    if features.ndim != 2 or len(features) != len(ids):
        raise ValueError("feature rows must match candidate IDs")
    if not np.isfinite(scores).all() or not np.isfinite(features).all():
        raise ValueError("scores and features must be finite")
    if not ids:
        return []
    span = np.ptp(scores)
    relevance = (scores - scores.min()) / span if span else np.zeros_like(scores)
    similarity = np.maximum(normalize(features) @ normalize(features).T, 0)
    remaining = list(range(len(ids)))
    selected = []
    while remaining and len(selected) < k:
        def objective(position):
            redundancy = similarity[position, selected].max() if selected else 0.0
            return (1 - diversity_weight) * relevance[position] - diversity_weight * redundancy
        best = max(remaining, key=objective)
        selected.append(best)
        remaining.remove(best)
    return [ids[position] for position in selected]
