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


def generate_catalog_dataset(
    n_users: int = 36,
    n_items: int = 60,
    interactions_per_user: int = 14,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create content-aligned histories for the advanced, still synthetic lab.

    All item metadata is known before interactions. The final catalog item has
    no interactions, making item cold-start observable. Histories are interleaved
    in time, with a preference for staying in the recently viewed genre. Each
    user rates an item at most once, so held-out positives remain recommendable.
    This deliberately simple world is not evidence about real-world accuracy.
    """

    if n_users < 1 or n_items < 6 or not 3 <= interactions_per_user < n_items:
        raise ValueError("need users >= 1, items >= 6, and 3 <= history length < items")
    rng = np.random.default_rng(seed)
    genres = ["Adventure", "Comedy", "Mystery", "Documentary"]
    keywords = ["冒険 宇宙 旅", "笑い 日常 友情", "謎 探偵 事件", "自然 科学 記録"]
    moods = ["穏やか", "刺激的", "感動"]
    item_genres = np.arange(n_items) % len(genres)
    items = pd.DataFrame(
        {
            "item_id": [f"i{idx + 1:03d}" for idx in range(n_items)],
            "title": [f"{genres[genre]} {idx + 1:03d}" for idx, genre in enumerate(item_genres)],
            "genre": [genres[genre] for genre in item_genres],
            "description": [
                f"{keywords[genre]} {moods[idx % len(moods)]}"
                for idx, genre in enumerate(item_genres)
            ],
        }
    )
    favorite_genres = rng.integers(0, len(genres), size=n_users)
    users = pd.DataFrame({
        "user_id": [f"u{idx + 1:03d}" for idx in range(n_users)],
        "favorite_genre": [genres[genre] for genre in favorite_genres],
    })
    rows = []
    for user_pos, user_id in enumerate(users["user_id"]):
        available = list(range(n_items - 1))
        last_genre = int(favorite_genres[user_pos])
        for step in range(interactions_per_user):
            candidate_genres = item_genres[available]
            taste = np.where(candidate_genres == favorite_genres[user_pos], 2.0, -0.4)
            recent = np.where(candidate_genres == last_genre, 0.8, 0.0)
            position = int(rng.choice(len(available), p=_softmax(taste + recent)))
            item_pos = available.pop(position)
            matches_taste = item_genres[item_pos] == favorite_genres[user_pos]
            noisy_rating = (4.7 if matches_taste else 2.8) + rng.normal(0, 0.6)
            rating = int(np.clip(np.rint(noisy_rating), 1, 5))
            rows.append({
                "user_id": user_id,
                "item_id": items.iloc[item_pos]["item_id"],
                "rating": rating,
                "timestamp": step * n_users + user_pos + 1,
            })
            last_genre = int(item_genres[item_pos])
    interactions = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return users, items, interactions
