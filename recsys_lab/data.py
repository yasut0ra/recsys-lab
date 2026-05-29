"""Dataset helpers for the recommender systems lab.

The goal of this file is not to create a realistic production dataset.
Instead, it creates a tiny interaction log that still contains useful
patterns: user taste, item popularity, noisy ratings, and timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


GENRES = [
    "Action",
    "Comedy",
    "Drama",
    "Documentary",
    "Fantasy",
    "Sci-Fi",
    "Sports",
]


def generate_dataset(
    n_users: int = 40,
    n_items: int = 70,
    min_interactions_per_user: int = 8,
    max_interactions_per_user: int = 18,
    n_factors: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a small explicit-feedback dataset.

    Returns:
        users: DataFrame with user metadata.
        items: DataFrame with item metadata.
        interactions: DataFrame with user-item ratings.

    Each user and item receives a hidden latent vector. We sample items with a
    probability related to the dot product between those hidden vectors, then
    convert that score into a 1-5 rating. This gives us data with structure
    while keeping the generator small enough to read in one sitting.
    """

    rng = np.random.default_rng(seed)

    user_ids = [f"u{idx:03d}" for idx in range(1, n_users + 1)]
    item_ids = [f"i{idx:03d}" for idx in range(1, n_items + 1)]

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "segment": rng.choice(["casual", "regular", "power"], size=n_users),
        }
    )

    items = pd.DataFrame(
        {
            "item_id": item_ids,
            "title": [f"Item {idx:03d}" for idx in range(1, n_items + 1)],
            "genre": rng.choice(GENRES, size=n_items),
        }
    )

    # Hidden vectors create taste structure. Item popularity makes some items
    # globally more likely to appear, which makes the popularity baseline useful.
    user_factors = rng.normal(0.0, 1.0, size=(n_users, n_factors))
    item_factors = rng.normal(0.0, 1.0, size=(n_items, n_factors))
    item_popularity = rng.normal(0.0, 0.7, size=n_items)

    rows: list[dict[str, object]] = []
    timestamp = 1

    for user_index, user_id in enumerate(user_ids):
        # Sample a different number of interactions for each user, just like a
        # real log where some users are much more active than others.
        n_user_interactions = int(
            rng.integers(min_interactions_per_user, max_interactions_per_user + 1)
        )
        n_user_interactions = min(n_user_interactions, n_items)

        raw_scores = item_factors @ user_factors[user_index] + item_popularity
        probabilities = _softmax(raw_scores)

        chosen_item_indices = rng.choice(
            np.arange(n_items),
            size=n_user_interactions,
            replace=False,
            p=probabilities,
        )

        for item_index in chosen_item_indices:
            noisy_score = raw_scores[item_index] + rng.normal(0.0, 0.8)

            # Map the noisy preference score to a familiar 1-5 explicit rating.
            rating = int(np.clip(np.rint(3.0 + noisy_score), 1, 5))

            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_ids[item_index],
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
            timestamp += 1

    interactions = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return users, items, interactions


def build_user_item_matrix(
    interactions: pd.DataFrame,
    value_col: str = "rating",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Convert an interaction log into a user-item matrix.

    Rows are users, columns are items, and missing interactions are filled with
    `fill_value`. A dense matrix is fine here because the dataset is tiny.
    Production systems usually use sparse matrices.
    """

    matrix = interactions.pivot_table(
        index="user_id",
        columns="item_id",
        values=value_col,
        aggfunc="mean",
    )
    matrix = matrix.sort_index().sort_index(axis=1)
    return matrix.fillna(fill_value)


def make_implicit_feedback(
    interactions: pd.DataFrame,
    rating_threshold: float = 4.0,
    alpha: float = 20.0,
) -> pd.DataFrame:
    """Turn explicit ratings into a small implicit-feedback table.

    `preference` is binary: 1 means we observed a positive signal. `confidence`
    says how strongly we trust that signal. This is a common way to explain
    implicit feedback: observed actions are positive, and missing actions are
    unknown rather than true negatives.
    """

    implicit = interactions.copy()
    implicit["preference"] = (implicit["rating"] >= rating_threshold).astype(int)
    implicit["confidence"] = 1.0 + alpha * implicit["preference"]
    return implicit


def _softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for sampling items."""

    shifted = values - values.max()
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()
