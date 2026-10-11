"""Small, dense reference models for the modern recommendation curriculum.

These implementations expose the math on tiny data. TF-IDF is not a pretrained
language model; the two towers are linear, retrieval is exact, and LightGCN
uses a dense propagation matrix. No external model downloads are needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _positive_rows(interactions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if "preference" in interactions:
        return interactions[interactions["preference"] > 0]
    return interactions[interactions["rating"] >= threshold]


class _RankingModel:
    """Shared ID alignment, positive indicators, exclusion, and fallback."""

    def _prepare(self, interactions, item_ids=None, threshold=4.0):
        if interactions.empty:
            raise ValueError("training interactions must not be empty")
        self.user_ids_ = np.array(sorted(interactions["user_id"].unique()))
        self.item_ids_ = np.array(
            sorted(interactions["item_id"].unique())
            if item_ids is None else list(item_ids)
        )
        if len(set(self.item_ids_)) != len(self.item_ids_):
            raise ValueError("item IDs must be unique")
        self.user_to_pos_ = {value: idx for idx, value in enumerate(self.user_ids_)}
        self.item_to_pos_ = {value: idx for idx, value in enumerate(self.item_ids_)}
        if not set(interactions["item_id"]).issubset(self.item_to_pos_):
            raise ValueError("training item is missing from the catalog")
        self.seen_items_ = interactions.groupby("user_id")["item_id"].agg(set).to_dict()
        self.preference_ = np.zeros((len(self.user_ids_), len(self.item_ids_)))
        for row in _positive_rows(interactions, threshold).itertuples(index=False):
            self.preference_[self.user_to_pos_[row.user_id], self.item_to_pos_[row.item_id]] = 1.0
        # Count observed records as in the original popularity baseline.
        counts = interactions.groupby("item_id").size()
        self.popularity_ = np.array(
            [counts.get(item, 0) for item in self.item_ids_], dtype=float
        )

    def recommend(self, user_id, k=10, exclude_seen=True):
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self.score_items(user_id)
        seen = self.seen_items_.get(user_id, set()) if exclude_seen else set()
        order = np.argsort(-scores, kind="stable")
        return [
            str(self.item_ids_[pos])
            for pos in order if self.item_ids_[pos] not in seen
        ][:k]

    def _has_profile(self, user_id):
        return user_id in self.user_to_pos_ and self.preference_[self.user_to_pos_[user_id]].any()


def _catalog_features(items: pd.DataFrame, text_col: str):
    if items.empty or items["item_id"].duplicated().any():
        raise ValueError("catalog must contain unique item IDs")
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    features = vectorizer.fit_transform(items[text_col].fillna("")).toarray()
    return vectorizer, features


@dataclass
class ContentBasedRecommender(_RankingModel):
    """Average the TF-IDF vectors of positively rated items."""

    rating_threshold: float = 4.0
    text_col: str = "description"

    def fit(self, interactions, items):
        self._prepare(interactions, items["item_id"], self.rating_threshold)
        self.vectorizer_, self.item_features_ = _catalog_features(items, self.text_col)
        self.user_profiles_ = normalize(self.preference_ @ self.item_features_)
        return self

    def score_items(self, user_id):
        if not self._has_profile(user_id):
            return self.popularity_.copy()
        return self.item_features_ @ self.user_profiles_[self.user_to_pos_[user_id]]


@dataclass
class EASERecommender(_RankingModel):
    """Closed-form ridge reconstruction with a zero coefficient diagonal."""

    regularization: float = 20.0
    rating_threshold: float = 4.0

    def fit(self, interactions, item_ids=None):
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        self._prepare(interactions, item_ids, self.rating_threshold)
        gram = self.preference_.T @ self.preference_
        inverse = np.linalg.solve(
            gram + self.regularization * np.eye(len(gram)), np.eye(len(gram))
        )
        self.coefficients_ = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.coefficients_, 0.0)
        return self

    def score_items(self, user_id):
        if not self._has_profile(user_id):
            return self.popularity_.copy()
        return self.preference_[self.user_to_pos_[user_id]] @ self.coefficients_


@dataclass
class ImplicitALSRecommender(_RankingModel):
    """Alternating weighted least squares, c_ui = 1 + alpha * p_ui."""

    n_factors: int = 8
    n_epochs: int = 10
    alpha: float = 20.0
    regularization: float = 0.1
    rating_threshold: float = 4.0
    seed: int = 42

    def fit(self, interactions, item_ids=None):
        if (
            self.n_factors < 1 or self.n_epochs < 1
            or self.alpha < 0 or self.regularization <= 0
        ):
            raise ValueError("invalid ALS dimensions, epochs, confidence, or regularization")
        self._prepare(interactions, item_ids, self.rating_threshold)
        rng = np.random.default_rng(self.seed)
        self.user_factors_ = rng.normal(0, 0.1, (len(self.user_ids_), self.n_factors))
        self.item_factors_ = rng.normal(0, 0.1, (len(self.item_ids_), self.n_factors))
        self.loss_history_ = []
        for _ in range(self.n_epochs):
            self.user_factors_ = self._solve(self.item_factors_, self.preference_)
            self.item_factors_ = self._solve(self.user_factors_, self.preference_.T)
            error = self.preference_ - self.user_factors_ @ self.item_factors_.T
            penalty = self.regularization * (
                np.square(self.user_factors_).sum()
                + np.square(self.item_factors_).sum()
            )
            weighted_error = ((1 + self.alpha * self.preference_) * error ** 2).sum()
            self.loss_history_.append(float(weighted_error + penalty))
        return self

    def _solve(self, fixed, preferences):
        result = np.zeros((len(preferences), self.n_factors))
        base = fixed.T @ fixed + self.regularization * np.eye(self.n_factors)
        for idx, preference in enumerate(preferences):
            confidence_extra = self.alpha * preference
            lhs = base + fixed.T @ (confidence_extra[:, None] * fixed)
            rhs = fixed.T @ ((1 + confidence_extra) * preference)
            result[idx] = np.linalg.solve(lhs, rhs)
        return result

    def score_items(self, user_id):
        if not self._has_profile(user_id):
            return self.popularity_.copy()
        return self.item_factors_ @ self.user_factors_[self.user_to_pos_[user_id]]


@dataclass
class LinearTwoTowerRecommender(_RankingModel):
    """Learn a user-ID tower and a TF-IDF item tower with full softmax.

    Other known positives are masked from each example's negative candidates.
    This exact, small-catalog loss is not sampled softmax or in-batch training.
    """

    n_factors: int = 12
    n_epochs: int = 60
    learning_rate: float = 0.3
    regularization: float = 0.001
    rating_threshold: float = 4.0
    seed: int = 42
    text_col: str = "description"

    def fit(self, interactions, items):
        if (
            self.n_factors < 1 or self.n_epochs < 1
            or self.learning_rate <= 0 or self.regularization < 0
        ):
            raise ValueError("invalid two-tower training parameters")
        self._prepare(interactions, items["item_id"], self.rating_threshold)
        self.vectorizer_, self.item_features_ = _catalog_features(items, self.text_col)
        rng = np.random.default_rng(self.seed)
        self.user_embeddings_ = rng.normal(0, 0.1, (len(self.user_ids_), self.n_factors))
        self.item_projection_ = rng.normal(0, 0.1, (self.item_features_.shape[1], self.n_factors))
        pairs = np.argwhere(self.preference_ > 0)
        if not len(pairs):
            raise ValueError("two-tower training needs a positive row")
        self.loss_history_ = []
        for _ in range(self.n_epochs):
            item_embeddings = self.item_features_ @ self.item_projection_
            user_gradient = np.zeros_like(self.user_embeddings_)
            item_gradient = np.zeros_like(item_embeddings)
            losses = []
            for user_pos, target in pairs:
                logits = item_embeddings @ self.user_embeddings_[user_pos]
                allowed = self.preference_[user_pos] == 0
                allowed[target] = True
                masked = np.where(allowed, logits, -np.inf)
                log_normalizer = np.logaddexp.reduce(masked)
                probabilities = np.exp(masked - log_normalizer)
                losses.append(log_normalizer - logits[target])
                probabilities[target] -= 1.0
                user_gradient[user_pos] += probabilities @ item_embeddings
                item_gradient += probabilities[:, None] * self.user_embeddings_[user_pos]
            self.loss_history_.append(float(np.mean(losses)))
            self.user_embeddings_ -= self.learning_rate * (
                user_gradient / len(pairs)
                + self.regularization * self.user_embeddings_
            )
            self.item_projection_ -= self.learning_rate * (
                self.item_features_.T @ item_gradient / len(pairs)
                + self.regularization * self.item_projection_
            )
        self.item_embeddings_ = self.item_features_ @ self.item_projection_
        return self

    def score_items(self, user_id):
        if not self._has_profile(user_id):
            return self.popularity_.copy()
        return self.item_embeddings_ @ self.user_embeddings_[self.user_to_pos_[user_id]]


@dataclass
class TransitionRecommender(_RankingModel):
    """A first-order sequence baseline; count consecutive positive events."""

    smoothing: float = 1.0
    rating_threshold: float = 4.0

    def fit(self, interactions, item_ids=None):
        if self.smoothing < 0:
            raise ValueError("smoothing must be nonnegative")
        self._prepare(interactions, item_ids, self.rating_threshold)
        self.transitions_ = np.zeros((len(self.item_ids_), len(self.item_ids_)))
        self.last_items_ = {}
        for user_id, rows in _positive_rows(interactions, self.rating_threshold).groupby("user_id"):
            history = rows.sort_values("timestamp", kind="stable")["item_id"].tolist()
            self.last_items_[user_id] = history[-1]
            for before, after in zip(history, history[1:]):
                self.transitions_[self.item_to_pos_[before], self.item_to_pos_[after]] += 1
        return self

    def score_items(self, user_id):
        if user_id not in self.last_items_:
            return self.popularity_.copy()
        prior = self.popularity_ / max(self.popularity_.sum(), 1)
        return self.transitions_[self.item_to_pos_[self.last_items_[user_id]]] + self.smoothing * prior


@dataclass
class LightGCNRecommender(_RankingModel):
    """Dense LightGCN propagation plus BPR gradients into initial embeddings."""

    n_factors: int = 12
    n_layers: int = 2
    n_epochs: int = 60
    learning_rate: float = 0.5
    regularization: float = 0.001
    rating_threshold: float = 4.0
    seed: int = 42

    def fit(self, interactions, item_ids=None):
        if (
            self.n_factors < 1 or self.n_layers < 0 or self.n_epochs < 1
            or self.learning_rate <= 0 or self.regularization < 0
        ):
            raise ValueError("invalid LightGCN training parameters")
        self._prepare(interactions, item_ids, self.rating_threshold)
        n_users, n_items = self.preference_.shape
        adjacency = np.zeros((n_users + n_items, n_users + n_items))
        adjacency[:n_users, n_users:] = self.preference_
        adjacency[n_users:, :n_users] = self.preference_.T
        degree = adjacency.sum(axis=1)
        inverse_degree = np.divide(
            1, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0
        )
        self.normalized_adjacency_ = (
            inverse_degree[:, None] * adjacency * inverse_degree[None, :]
        )
        power = np.eye(len(adjacency))
        self.propagation_ = power.copy()
        for _ in range(self.n_layers):
            power = power @ self.normalized_adjacency_
            self.propagation_ += power
        self.propagation_ /= self.n_layers + 1
        rng = np.random.default_rng(self.seed)
        self.initial_embeddings_ = rng.normal(0, 0.1, (len(adjacency), self.n_factors))
        negatives = [np.flatnonzero(row == 0) for row in self.preference_]
        pairs = [(u, i) for u, i in np.argwhere(self.preference_ > 0) if len(negatives[u])]
        self.loss_history_ = []
        for _ in range(self.n_epochs):
            embeddings = self.propagation_ @ self.initial_embeddings_
            gradient = np.zeros_like(embeddings)
            losses = []
            for user_pos, positive_pos in pairs:
                negative_pos = int(rng.choice(negatives[user_pos]))
                positive_node, negative_node = n_users + positive_pos, n_users + negative_pos
                difference = embeddings[positive_node] - embeddings[negative_node]
                margin = float(embeddings[user_pos] @ difference)
                scale = float(np.exp(-np.logaddexp(0, margin)))
                losses.append(np.logaddexp(0, -margin))
                gradient[user_pos] -= scale * difference
                gradient[positive_node] -= scale * embeddings[user_pos]
                gradient[negative_node] += scale * embeddings[user_pos]
            divisor = max(len(pairs), 1)
            initial_gradient = self.propagation_.T @ gradient / divisor
            self.initial_embeddings_ -= self.learning_rate * (
                initial_gradient + self.regularization * self.initial_embeddings_
            )
            self.loss_history_.append(float(np.mean(losses)) if losses else 0.0)
        embeddings = self.propagation_ @ self.initial_embeddings_
        self.user_embeddings_, self.item_embeddings_ = embeddings[:n_users], embeddings[n_users:]
        return self

    def score_items(self, user_id):
        if not self._has_profile(user_id):
            return self.popularity_.copy()
        return self.item_embeddings_ @ self.user_embeddings_[self.user_to_pos_[user_id]]


@dataclass
class HybridRRFRecommender:
    """Combine fitted model rankings with reciprocal rank fusion (RRF)."""

    models: list
    rank_constant: float = 60.0

    def recommend(self, user_id, k=10, exclude_seen=True):
        if k <= 0 or self.rank_constant <= 0 or not self.models:
            raise ValueError("need positive k, constant, and at least one model")
        scores = {}
        for model in self.models:
            size = len(model.item_ids_)
            ranking = model.recommend(user_id, k=size, exclude_seen=exclude_seen)
            for rank, item_id in enumerate(ranking, start=1):
                scores[item_id] = scores.get(item_id, 0.0) + 1 / (self.rank_constant + rank)
        return sorted(scores, key=lambda item: (-scores[item], item))[:k]
