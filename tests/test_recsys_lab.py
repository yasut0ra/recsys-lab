import unittest

from recsys_lab.data import generate_dataset, make_implicit_feedback
from recsys_lab.metrics import ndcg_at_k, precision_at_k, recall_at_k, evaluate_at_k
from recsys_lab.models import (
    ImplicitBPRRecommender,
    ItemBasedCFRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender,
    UserBasedCFRecommender,
)
from recsys_lab.split import train_test_split_by_user


class RecsysLabTest(unittest.TestCase):
    def setUp(self):
        self.users, self.items, self.interactions = generate_dataset(
            n_users=12,
            n_items=24,
            min_interactions_per_user=5,
            max_interactions_per_user=8,
            seed=7,
        )
        self.train, self.test = train_test_split_by_user(
            self.interactions,
            test_ratio=0.25,
            seed=7,
        )

    def test_dataset_has_expected_columns(self):
        self.assertEqual(
            set(["user_id", "item_id", "rating", "timestamp"]),
            set(self.interactions.columns),
        )
        self.assertTrue(self.interactions["rating"].between(1, 5).all())

    def test_split_has_no_overlap(self):
        train_keys = set(zip(self.train["user_id"], self.train["item_id"]))
        test_keys = set(zip(self.test["user_id"], self.test["item_id"]))
        self.assertTrue(train_keys.isdisjoint(test_keys))
        self.assertGreater(len(self.train), len(self.test))

    def test_metrics_known_values(self):
        recommended = ["a", "b", "c", "d"]
        relevant = {"b", "d", "x"}

        self.assertAlmostEqual(precision_at_k(recommended, relevant, k=4), 0.5)
        self.assertAlmostEqual(recall_at_k(recommended, relevant, k=4), 2 / 3)
        self.assertGreater(ndcg_at_k(recommended, relevant, k=4), 0.0)
        self.assertLessEqual(ndcg_at_k(recommended, relevant, k=4), 1.0)

    def test_recommenders_return_unseen_items(self):
        models = [
            PopularityRecommender(),
            UserBasedCFRecommender(k_neighbors=4),
            ItemBasedCFRecommender(k_neighbors=4),
            MatrixFactorizationRecommender(n_factors=4, n_epochs=3, seed=7),
        ]

        user_id = self.train["user_id"].iloc[0]
        seen = set(self.train[self.train["user_id"] == user_id]["item_id"])

        for model in models:
            model.fit(self.train)
            recommended = model.recommend(user_id, k=3, exclude_seen=True)
            self.assertLessEqual(len(recommended), 3)
            self.assertTrue(set(recommended).isdisjoint(seen))

    def test_implicit_bpr_runs(self):
        implicit_train = make_implicit_feedback(self.train, rating_threshold=4.0)
        model = ImplicitBPRRecommender(n_factors=4, n_epochs=3, seed=7)
        model.fit(implicit_train)

        user_id = implicit_train["user_id"].iloc[0]
        recommended = model.recommend(user_id, k=3, exclude_seen=True)
        self.assertLessEqual(len(recommended), 3)

    def test_evaluate_at_k_returns_metric_dict(self):
        model = PopularityRecommender().fit(self.train)
        metrics = evaluate_at_k(model, self.train, self.test, k=5)

        self.assertEqual(set(metrics), {"precision", "recall", "ndcg", "n_users"})
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertLessEqual(metrics["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
