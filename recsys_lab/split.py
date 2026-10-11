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


def temporal_train_validation_test_split(
    interactions: pd.DataFrame,
    validation_size: int = 1,
    test_size: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep each user's latest timestamp groups for validation and test.

    Sizes count distinct timestamps, so tied events never cross partitions.
    Users need at least one earlier timestamp group for training; shorter
    histories stay entirely in train. This is per-user leave-last-out, not a
    global time cutoff: another user's training events may occur later.
    """

    if validation_size < 1 or test_size < 1:
        raise ValueError("validation_size and test_size must be positive")
    if interactions["timestamp"].isna().any():
        raise ValueError("timestamps must not be missing")
    train_parts, validation_parts, test_parts = [], [], []
    for _, rows in interactions.groupby("user_id", sort=False):
        rows = rows.sort_values("timestamp", kind="stable")
        times = rows["timestamp"].drop_duplicates().to_numpy()
        if len(times) <= validation_size + test_size:
            train_parts.append(rows)
            continue
        validation_start = times[-(validation_size + test_size)]
        test_start = times[-test_size]
        train_parts.append(rows[rows["timestamp"] < validation_start])
        validation_parts.append(rows[
            (rows["timestamp"] >= validation_start)
            & (rows["timestamp"] < test_start)
        ])
        test_parts.append(rows[rows["timestamp"] >= test_start])

    def combine(parts):
        if not parts:
            return interactions.iloc[:0].copy()
        return pd.concat(parts).sort_values("timestamp", kind="stable").reset_index(drop=True)

    return combine(train_parts), combine(validation_parts), combine(test_parts)
