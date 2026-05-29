"""Educational recommender implementations.

The models in this file are intentionally small and explicit. They are useful
for learning the core ideas, not for serving large-scale production traffic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recsys_lab.data import build_user_item_matrix


def _seen_items_by_user(interactions: pd.DataFrame) -> dict[str, set[str]]:
    """Create a lookup table used to avoid recommending already-seen items."""

    return (
        interactions.groupby("user_id")["item_id"]
        .apply(lambda values: set(values))
        .to_dict()
    )


class PopularityRecommender:
    """Recommend globally popular items.

    This is the baseline every recommender project should start with. If a more
    complex model cannot beat popularity, the extra complexity is not justified.
    """

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        item_stats = (
            interactions.groupby("item_id")
            .agg(count=("rating", "size"), mean_rating=("rating", "mean"))
            .reset_index()
        )

        # Count is the main popularity signal. Mean rating breaks ties.
        item_stats = item_stats.sort_values(
            ["count", "mean_rating", "item_id"],
            ascending=[False, False, True],
        )

        self.ranked_items_ = item_stats["item_id"].tolist()
        self.item_scores_ = dict(zip(item_stats["item_id"], item_stats["count"]))
        self.seen_items_ = _seen_items_by_user(interactions)
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[str]:
        seen = self.seen_items_.get(user_id, set()) if exclude_seen else set()
        return [item_id for item_id in self.ranked_items_ if item_id not in seen][:k]


@dataclass
class UserBasedCFRecommender:
    """User-based collaborative filtering with cosine similarity."""

    k_neighbors: int = 10

    def fit(self, interactions: pd.DataFrame) -> "UserBasedCFRecommender":
        self.matrix_ = build_user_item_matrix(interactions)
        self.user_ids_ = self.matrix_.index.to_numpy()
        self.item_ids_ = self.matrix_.columns.to_numpy()
        self.seen_items_ = _seen_items_by_user(interactions)

        ratings = self.matrix_.to_numpy(dtype=float)
        observed = ratings > 0

        # Users have different rating habits. Centering by each user's mean
        # reduces the effect of "generous" and "strict" raters.
        counts = observed.sum(axis=1)
        sums = ratings.sum(axis=1)
        self.user_means_ = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts > 0,
        )

        centered = np.where(observed, ratings - self.user_means_[:, None], 0.0)
        self.centered_matrix_ = centered
        self.similarity_ = cosine_similarity(centered)

        self.fallback_ = PopularityRecommender().fit(interactions)
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[str]:
        if user_id not in self.matrix_.index:
            return self.fallback_.recommend(user_id, k=k, exclude_seen=exclude_seen)

        user_pos = int(np.where(self.user_ids_ == user_id)[0][0])
        similarities = self.similarity_[user_pos].copy()
        similarities[user_pos] = 0.0

        # Negative similarity usually means opposite taste, so this simple
        # version keeps only positive neighbors.
        neighbor_positions = np.argsort(similarities)[::-1]
        neighbor_positions = [
            pos for pos in neighbor_positions if similarities[pos] > 0
        ][: self.k_neighbors]

        if not neighbor_positions:
            return self.fallback_.recommend(user_id, k=k, exclude_seen=exclude_seen)

        neighbor_sims = similarities[neighbor_positions]
        neighbor_centered = self.centered_matrix_[neighbor_positions]

        numerator = neighbor_sims @ neighbor_centered
        denominator = np.abs(neighbor_sims) @ (neighbor_centered != 0)
        scores = self.user_means_[user_pos] + np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )

        return _top_k_items(
            scores=scores,
            item_ids=self.item_ids_,
            seen_items=self.seen_items_.get(user_id, set()) if exclude_seen else set(),
            k=k,
        )


@dataclass
class ItemBasedCFRecommender:
    """Item-based collaborative filtering with cosine similarity."""

    k_neighbors: int = 20

    def fit(self, interactions: pd.DataFrame) -> "ItemBasedCFRecommender":
        self.matrix_ = build_user_item_matrix(interactions)
        self.user_ids_ = self.matrix_.index.to_numpy()
        self.item_ids_ = self.matrix_.columns.to_numpy()
        self.seen_items_ = _seen_items_by_user(interactions)

        ratings = self.matrix_.to_numpy(dtype=float)
        observed = ratings > 0

        # Centering by item mean makes "universally high-rated" items less able
        # to dominate similarity purely because many people liked them.
        counts = observed.sum(axis=0)
        sums = ratings.sum(axis=0)
        self.item_means_ = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts > 0,
        )

        centered = np.where(observed, ratings - self.item_means_[None, :], 0.0)
        self.similarity_ = cosine_similarity(centered.T)

        self.fallback_ = PopularityRecommender().fit(interactions)
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[str]:
        if user_id not in self.matrix_.index:
            return self.fallback_.recommend(user_id, k=k, exclude_seen=exclude_seen)

        user_pos = int(np.where(self.user_ids_ == user_id)[0][0])
        user_ratings = self.matrix_.to_numpy(dtype=float)[user_pos]
        seen_positions = np.where(user_ratings > 0)[0]

        if len(seen_positions) == 0:
            return self.fallback_.recommend(user_id, k=k, exclude_seen=exclude_seen)

        scores = np.zeros(len(self.item_ids_), dtype=float)

        for item_pos in range(len(self.item_ids_)):
            sims = self.similarity_[item_pos, seen_positions].copy()
            ratings = user_ratings[seen_positions]

            # Use only the most similar items the user has already rated.
            order = np.argsort(sims)[::-1]
            order = [idx for idx in order if sims[idx] > 0][: self.k_neighbors]

            if order:
                chosen_sims = sims[order]
                chosen_ratings = ratings[order]
                scores[item_pos] = np.dot(chosen_sims, chosen_ratings) / (
                    np.abs(chosen_sims).sum() + 1e-12
                )
            else:
                scores[item_pos] = self.item_means_[item_pos]

        return _top_k_items(
            scores=scores,
            item_ids=self.item_ids_,
            seen_items=self.seen_items_.get(user_id, set()) if exclude_seen else set(),
            k=k,
        )


@dataclass
class MatrixFactorizationRecommender:
    """Explicit-feedback matrix factorization trained with SGD."""

    n_factors: int = 16
    n_epochs: int = 30
    learning_rate: float = 0.03
    regularization: float = 0.04
    seed: int = 42

    def fit(self, interactions: pd.DataFrame) -> "MatrixFactorizationRecommender":
        rng = np.random.default_rng(self.seed)

        self.user_ids_ = np.array(sorted(interactions["user_id"].unique()))
        self.item_ids_ = np.array(sorted(interactions["item_id"].unique()))
        self.user_to_pos_ = {user_id: pos for pos, user_id in enumerate(self.user_ids_)}
        self.item_to_pos_ = {item_id: pos for pos, item_id in enumerate(self.item_ids_)}
        self.seen_items_ = _seen_items_by_user(interactions)

        n_users = len(self.user_ids_)
        n_items = len(self.item_ids_)

        self.global_mean_ = float(interactions["rating"].mean())
        self.user_bias_ = np.zeros(n_users, dtype=float)
        self.item_bias_ = np.zeros(n_items, dtype=float)
        self.user_factors_ = rng.normal(0.0, 0.1, size=(n_users, self.n_factors))
        self.item_factors_ = rng.normal(0.0, 0.1, size=(n_items, self.n_factors))

        samples = [
            (
                self.user_to_pos_[row.user_id],
                self.item_to_pos_[row.item_id],
                float(row.rating),
            )
            for row in interactions.itertuples(index=False)
        ]

        for _ in range(self.n_epochs):
            rng.shuffle(samples)

            for user_pos, item_pos, rating in samples:
                prediction = self._predict_pos(user_pos, item_pos)
                error = rating - prediction

                user_vector = self.user_factors_[user_pos].copy()
                item_vector = self.item_factors_[item_pos].copy()

                # Bias terms learn that some users rate high and some items are
                # broadly loved. Factor vectors learn user-item interactions.
                self.user_bias_[user_pos] += self.learning_rate * (
                    error - self.regularization * self.user_bias_[user_pos]
                )
                self.item_bias_[item_pos] += self.learning_rate * (
                    error - self.regularization * self.item_bias_[item_pos]
                )
                self.user_factors_[user_pos] += self.learning_rate * (
                    error * item_vector - self.regularization * user_vector
                )
                self.item_factors_[item_pos] += self.learning_rate * (
                    error * user_vector - self.regularization * item_vector
                )

        self.fallback_ = PopularityRecommender().fit(interactions)
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[str]:
        if user_id not in self.user_to_pos_:
            return self.fallback_.recommend(user_id, k=k, exclude_seen=exclude_seen)

        user_pos = self.user_to_pos_[user_id]
        scores = (
            self.global_mean_
            + self.user_bias_[user_pos]
            + self.item_bias_
            + self.item_factors_ @ self.user_factors_[user_pos]
        )

        return _top_k_items(
            scores=scores,
            item_ids=self.item_ids_,
            seen_items=self.seen_items_.get(user_id, set()) if exclude_seen else set(),
            k=k,
        )

    def _predict_pos(self, user_pos: int, item_pos: int) -> float:
        return float(
            self.global_mean_
            + self.user_bias_[user_pos]
            + self.item_bias_[item_pos]
            + np.dot(self.user_factors_[user_pos], self.item_factors_[item_pos])
        )


@dataclass
class ImplicitBPRRecommender:
    """Implicit-feedback matrix factorization with BPR-style training.

    BPR learns from triples `(user, positive_item, negative_item)`. The model is
    updated so the observed positive item receives a higher score than a sampled
    unobserved item.
    """

    n_factors: int = 16
    n_epochs: int = 30
    learning_rate: float = 0.05
    regularization: float = 0.01
    seed: int = 42

    def fit(self, interactions: pd.DataFrame) -> "ImplicitBPRRecommender":
        rng = np.random.default_rng(self.seed)

        # If a preference column exists, use positive rows only. Otherwise, every
        # observed interaction is treated as a positive implicit signal.
        if "preference" in interactions.columns:
            positives = interactions[interactions["preference"] > 0].copy()
        else:
            positives = interactions.copy()

        if positives.empty:
            raise ValueError("ImplicitBPRRecommender needs at least one positive row")

        self.user_ids_ = np.array(sorted(positives["user_id"].unique()))
        self.item_ids_ = np.array(sorted(interactions["item_id"].unique()))
        self.user_to_pos_ = {user_id: pos for pos, user_id in enumerate(self.user_ids_)}
        self.item_to_pos_ = {item_id: pos for pos, item_id in enumerate(self.item_ids_)}
        self.seen_items_ = _seen_items_by_user(interactions)

        n_users = len(self.user_ids_)
        n_items = len(self.item_ids_)

        self.user_factors_ = rng.normal(0.0, 0.1, size=(n_users, self.n_factors))
        self.item_factors_ = rng.normal(0.0, 0.1, size=(n_items, self.n_factors))
        self.item_bias_ = np.zeros(n_items, dtype=float)

        positives_by_user = (
            positives.groupby("user_id")["item_id"]
            .apply(lambda values: set(values))
            .to_dict()
        )

        training_pairs = [
            (self.user_to_pos_[row.user_id], self.item_to_pos_[row.item_id])
            for row in positives.itertuples(index=False)
        ]

        all_item_positions = np.arange(n_items)

        for _ in range(self.n_epochs):
            rng.shuffle(training_pairs)

            for user_pos, positive_pos in training_pairs:
                user_id = self.user_ids_[user_pos]
                positive_items = positives_by_user[user_id]

                # Sample an item with no observed positive signal for this user.
                negative_pos = _sample_negative_item(
                    rng=rng,
                    all_item_positions=all_item_positions,
                    item_ids=self.item_ids_,
                    positive_items=positive_items,
                )

                self._update_pair(user_pos, positive_pos, negative_pos)

        self.fallback_ = PopularityRecommender().fit(interactions)
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[str]:
        if user_id not in self.user_to_pos_:
            return self.fallback_.recommend(user_id, k=k, exclude_seen=exclude_seen)

        user_pos = self.user_to_pos_[user_id]
        scores = self.item_bias_ + self.item_factors_ @ self.user_factors_[user_pos]

        return _top_k_items(
            scores=scores,
            item_ids=self.item_ids_,
            seen_items=self.seen_items_.get(user_id, set()) if exclude_seen else set(),
            k=k,
        )

    def _update_pair(
        self,
        user_pos: int,
        positive_pos: int,
        negative_pos: int,
    ) -> None:
        user_vector = self.user_factors_[user_pos].copy()
        positive_vector = self.item_factors_[positive_pos].copy()
        negative_vector = self.item_factors_[negative_pos].copy()

        positive_score = self.item_bias_[positive_pos] + np.dot(
            user_vector, positive_vector
        )
        negative_score = self.item_bias_[negative_pos] + np.dot(
            user_vector, negative_vector
        )

        # sigmoid(-x) is the gradient scale for maximizing log sigmoid(x).
        score_diff = positive_score - negative_score
        gradient_scale = 1.0 / (1.0 + math.exp(score_diff))

        self.user_factors_[user_pos] += self.learning_rate * (
            gradient_scale * (positive_vector - negative_vector)
            - self.regularization * user_vector
        )
        self.item_factors_[positive_pos] += self.learning_rate * (
            gradient_scale * user_vector - self.regularization * positive_vector
        )
        self.item_factors_[negative_pos] += self.learning_rate * (
            -gradient_scale * user_vector - self.regularization * negative_vector
        )
        self.item_bias_[positive_pos] += self.learning_rate * (
            gradient_scale - self.regularization * self.item_bias_[positive_pos]
        )
        self.item_bias_[negative_pos] += self.learning_rate * (
            -gradient_scale - self.regularization * self.item_bias_[negative_pos]
        )


def _sample_negative_item(
    rng: np.random.Generator,
    all_item_positions: np.ndarray,
    item_ids: np.ndarray,
    positive_items: set[str],
) -> int:
    """Sample an item that is not positive for the current user."""

    while True:
        item_pos = int(rng.choice(all_item_positions))
        if item_ids[item_pos] not in positive_items:
            return item_pos


def _top_k_items(
    scores: np.ndarray,
    item_ids: np.ndarray,
    seen_items: set[str],
    k: int,
) -> list[str]:
    """Return item ids sorted by score while filtering seen items."""

    ranked_positions = np.argsort(scores)[::-1]
    recommendations: list[str] = []

    for item_pos in ranked_positions:
        item_id = str(item_ids[item_pos])
        if item_id in seen_items:
            continue
        recommendations.append(item_id)
        if len(recommendations) == k:
            break

    return recommendations
