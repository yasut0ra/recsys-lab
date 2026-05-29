"""Ranking metrics for top-K recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(
    recommended_items: list[str],
    relevant_items: set[str],
    k: int,
) -> float:
    """Fraction of the top-K recommendation list that is relevant."""

    if k <= 0:
        raise ValueError("k must be positive")

    top_k = recommended_items[:k]
    if not top_k:
        return 0.0

    hits = len(set(top_k) & relevant_items)
    return hits / k


def recall_at_k(
    recommended_items: list[str],
    relevant_items: set[str],
    k: int,
) -> float:
    """Fraction of all relevant items recovered in the top-K list."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_items:
        return 0.0

    top_k = recommended_items[:k]
    hits = len(set(top_k) & relevant_items)
    return hits / len(relevant_items)


def ndcg_at_k(
    recommended_items: list[str],
    relevant_items: set[str],
    k: int,
) -> float:
    """Normalized discounted cumulative gain at K.

    Binary relevance is used here: an item is either relevant or not. A hit at
    rank 1 is worth more than a hit at rank 5.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_items:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(recommended_items[:k], start=1):
        if item_id in relevant_items:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_at_k(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    rating_threshold: float = 4.0,
) -> dict[str, float]:
    """Evaluate a fitted recommender with Precision@K, Recall@K, and NDCG@K.

    Test interactions with rating >= `rating_threshold` are treated as relevant.
    Users with no relevant test items are skipped because recall and NDCG would
    not have a useful denominator.
    """

    rows: list[dict[str, float]] = []

    relevant_by_user = (
        test[test["rating"] >= rating_threshold]
        .groupby("user_id")["item_id"]
        .apply(lambda values: set(values))
    )

    for user_id, relevant_items in relevant_by_user.items():
        recommended = model.recommend(user_id, k=k, exclude_seen=True)
        rows.append(
            {
                "precision": precision_at_k(recommended, relevant_items, k),
                "recall": recall_at_k(recommended, relevant_items, k),
                "ndcg": ndcg_at_k(recommended, relevant_items, k),
            }
        )

    if not rows:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "n_users": 0}

    frame = pd.DataFrame(rows)
    return {
        "precision": float(frame["precision"].mean()),
        "recall": float(frame["recall"].mean()),
        "ndcg": float(frame["ndcg"].mean()),
        "n_users": int(len(frame)),
    }
