"""Train/test split helpers.

For recommender systems, a split should preserve each user's training history.
If a user appears only in test, most collaborative recommenders cannot produce
personalized recommendations for that user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_split_by_user(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions user by user.

    Users with only one interaction are kept in train. For other users, at
    least one interaction remains in train and at least one goes to test.
    """

    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, user_rows in interactions.groupby("user_id", sort=False):
        user_rows = user_rows.sort_values("timestamp")

        if len(user_rows) < 2:
            train_parts.append(user_rows)
            continue

        n_test = max(1, int(round(len(user_rows) * test_ratio)))
        n_test = min(n_test, len(user_rows) - 1)

        shuffled_positions = rng.permutation(len(user_rows))
        test_positions = set(shuffled_positions[:n_test])

        is_test = np.array(
            [position in test_positions for position in range(len(user_rows))]
        )

        train_parts.append(user_rows.loc[~is_test])
        test_parts.append(user_rows.loc[is_test])

    train = pd.concat(train_parts).sort_values("timestamp").reset_index(drop=True)
    test = pd.concat(test_parts).sort_values("timestamp").reset_index(drop=True)
    return train, test
