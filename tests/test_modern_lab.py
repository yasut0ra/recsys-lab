import unittest

import numpy as np
import pandas as pd

from recsys_lab.data import generate_catalog_dataset
from recsys_lab.metrics import catalog_coverage, intra_list_diversity
from recsys_lab.modern_models import (
    ContentBasedRecommender, EASERecommender, HybridRRFRecommender,
    ImplicitALSRecommender, LightGCNRecommender, LinearTwoTowerRecommender,
    TransitionRecommender,
)
from recsys_lab.reranking import rerank_mmr
from recsys_lab.split import temporal_train_validation_test_split


class ModernLabTest(unittest.TestCase):
    def setUp(self):
        self.users, self.items, self.interactions = generate_catalog_dataset(
            n_users=8, n_items=20, interactions_per_user=8, seed=5
        )
        self.train, self.validation, self.test = temporal_train_validation_test_split(self.interactions)

    def test_temporal_split_keeps_tied_events_together_and_handles_short_histories(self):
        rows = pd.DataFrame({"user_id": ["a"] * 6 + ["b"], "item_id": list("ABCDEFG"),
                             "rating": [5] * 7, "timestamp": [1, 1, 2, 3, 4, 4, 2]})
        train, validation, test = temporal_train_validation_test_split(rows)
        self.assertEqual(set(train[train.user_id == "a"].timestamp), {1, 2})
        self.assertEqual(set(validation.timestamp), {3})
        self.assertEqual(set(test.timestamp), {4})
        self.assertIn("b", set(train.user_id))
        self.assertEqual(len(train) + len(validation) + len(test), len(rows))
        empty = rows.iloc[:0]
        self.assertTrue(all(part.empty for part in temporal_train_validation_test_split(empty)))

    def test_temporal_split_does_not_mix_future_events_for_a_user(self):
        for user in self.users.user_id:
            train_end = self.train[self.train.user_id == user].timestamp.max()
            validation_rows = self.validation[self.validation.user_id == user]
            test_rows = self.test[self.test.user_id == user]
            self.assertLess(train_end, validation_rows.timestamp.min())
            self.assertLess(validation_rows.timestamp.max(), test_rows.timestamp.min())

    def test_catalog_is_reproducible_has_cold_item_and_aligned_content(self):
        _, items, interactions = generate_catalog_dataset(8, 20, 8, seed=5)
        pd.testing.assert_frame_equal(items, self.items)
        pd.testing.assert_frame_equal(interactions, self.interactions)
        self.assertNotIn(self.items.iloc[-1].item_id, set(self.interactions.item_id))
        self.assertFalse(self.interactions.duplicated(["user_id", "item_id"]).any())
        self.assertTrue(self.items.description.str.len().gt(0).all())

    def test_content_can_recommend_a_catalog_item_without_interactions(self):
        items = pd.DataFrame({"item_id": ["seen", "new", "other"],
                              "description": ["宇宙 冒険", "宇宙 冒険", "笑い 日常"]})
        rows = pd.DataFrame({"user_id": ["u"], "item_id": ["seen"], "rating": [5]})
        model = ContentBasedRecommender().fit(rows, items)
        self.assertEqual(model.recommend("u", k=1), ["new"])
        self.assertEqual(model.recommend("unknown", k=1), ["seen"])

    def test_ease_matches_independent_constrained_ridge_solutions(self):
        model = EASERecommender(regularization=2).fit(self.train)
        x = model.preference_
        for column in range(x.shape[1]):
            others = np.arange(x.shape[1]) != column
            features = x[:, others]
            expected = np.linalg.solve(
                features.T @ features + 2 * np.eye(features.shape[1]),
                features.T @ x[:, column],
            )
            np.testing.assert_allclose(model.coefficients_[others, column], expected, atol=1e-12)
        np.testing.assert_array_equal(np.diag(model.coefficients_), 0)

    def test_als_weighted_objective_decreases(self):
        model = ImplicitALSRecommender(n_factors=3, n_epochs=8, seed=5).fit(self.train)
        self.assertTrue(np.isfinite(model.loss_history_).all())
        self.assertTrue(np.all(np.diff(model.loss_history_) <= 1e-8))
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_two_tower_learns_and_item_vectors_use_content(self):
        model = LinearTwoTowerRecommender(n_factors=4, n_epochs=60, seed=5).fit(self.train, self.items)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])
        np.testing.assert_allclose(model.item_embeddings_, model.item_features_ @ model.item_projection_)
        self.assertIn(self.items.iloc[-1].item_id, model.item_to_pos_)

    def test_lightgcn_gradient_matches_finite_difference(self):
        seed = 7
        rate = 0.2
        model = LightGCNRecommender(n_factors=3, n_layers=2, n_epochs=1,
                                    learning_rate=rate, regularization=0, seed=seed).fit(self.train)
        rng = np.random.default_rng(seed)
        initial = rng.normal(0, 0.1, model.initial_embeddings_.shape)
        n_users = len(model.user_ids_)
        negatives = [np.flatnonzero(row == 0) for row in model.preference_]
        pairs = [
            (u, i, int(rng.choice(negatives[u])))
            for u, i in np.argwhere(model.preference_ > 0) if len(negatives[u])
        ]

        def loss(parameters):
            vectors = model.propagation_ @ parameters
            margins = [vectors[u] @ (vectors[n_users + i] - vectors[n_users + j]) for u, i, j in pairs]
            return np.logaddexp(0, -np.array(margins)).mean()

        actual_gradient = (initial - model.initial_embeddings_) / rate
        epsilon = 1e-5
        for row, column in [(0, 0), (n_users, 1), (len(initial) - 1, 2)]:
            plus, minus = initial.copy(), initial.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            expected = (loss(plus) - loss(minus)) / (2 * epsilon)
            self.assertAlmostEqual(actual_gradient[row, column], expected, places=8)

    def test_sequence_direction_changes_recommendations(self):
        rows = pd.DataFrame({"user_id": ["a", "a", "b", "b", "query"],
                             "item_id": ["A", "B", "A", "C", "A"],
                             "rating": [5] * 5, "timestamp": [1, 2, 1, 2, 1]})
        model = TransitionRecommender(smoothing=0).fit(rows)
        self.assertEqual(model.recommend("query", k=2), ["B", "C"])
        reversed_rows = rows.copy()
        reversed_rows.loc[reversed_rows.user_id == "a", "timestamp"] = [2, 1]
        reversed_model = TransitionRecommender(smoothing=0).fit(reversed_rows)
        self.assertEqual(reversed_model.recommend("query", k=1), ["C"])

    def test_models_exclude_seen_and_handle_cold_users_and_saturated_histories(self):
        models = [ContentBasedRecommender().fit(self.train, self.items),
                  EASERecommender().fit(self.train),
                  ImplicitALSRecommender(n_epochs=2).fit(self.train),
                  LinearTwoTowerRecommender(n_epochs=2).fit(self.train, self.items),
                  TransitionRecommender().fit(self.train),
                  LightGCNRecommender(n_epochs=2).fit(self.train)]
        models.append(HybridRRFRecommender(models[:2]))
        for model in models:
            seen = set(self.train[self.train.user_id == "u001"].item_id)
            recommendation = model.recommend("u001", k=4)
            self.assertEqual(len(recommendation), len(set(recommendation)))
            self.assertTrue(seen.isdisjoint(recommendation))
            self.assertTrue(model.recommend("cold", k=3))
            with self.assertRaises(ValueError):
                model.recommend("u001", k=0)
        saturated = pd.DataFrame({"user_id": ["u", "u"], "item_id": ["a", "b"], "rating": [5, 5]})
        self.assertEqual(LightGCNRecommender(n_epochs=2).fit(saturated).recommend("u"), [])

    def test_mmr_trades_relevance_for_diversity_without_duplicates(self):
        ids, scores = ["a", "b", "c"], [1, 0.9, 0.8]
        features = np.array([[1, 0], [1, 0], [0, 1]])
        self.assertEqual(rerank_mmr(ids, scores, features, k=2, diversity_weight=0), ["a", "b"])
        self.assertEqual(rerank_mmr(ids, scores, features, k=2, diversity_weight=0.6), ["a", "c"])
        self.assertEqual(len(rerank_mmr(ids, scores, features, k=10)), 3)
        self.assertEqual(rerank_mmr([], [], np.empty((0, 2))), [])

    def test_coverage_and_diversity_known_values(self):
        self.assertEqual(catalog_coverage([["a", "a"], ["b", "foreign"]], ["a", "b", "c", "d"]), 0.5)
        self.assertEqual(intra_list_diversity(np.array([[1, 0], [1, 0]])), 0)
        self.assertEqual(intra_list_diversity(np.eye(2)), 1)


if __name__ == "__main__":
    unittest.main()
