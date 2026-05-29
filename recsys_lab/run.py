"""Run the full recommender-systems lab demo."""

from __future__ import annotations

import pandas as pd

from recsys_lab.data import generate_dataset, make_implicit_feedback
from recsys_lab.metrics import evaluate_at_k
from recsys_lab.models import (
    ImplicitBPRRecommender,
    ItemBasedCFRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender,
    UserBasedCFRecommender,
)
from recsys_lab.split import train_test_split_by_user


def main() -> None:
    users, items, interactions = generate_dataset(seed=42)
    train, test = train_test_split_by_user(interactions, test_ratio=0.2, seed=42)
    implicit_train = make_implicit_feedback(train, rating_threshold=4.0)

    print("=== Dataset ===")
    print(f"users:        {len(users)}")
    print(f"items:        {len(items)}")
    print(f"interactions: {len(interactions)}")
    print(f"train/test:   {len(train)} / {len(test)}")
    print()
    print(interactions.head(8).to_string(index=False))
    print()

    models = [
        ("Popularity", PopularityRecommender(), train),
        ("User-based CF", UserBasedCFRecommender(k_neighbors=10), train),
        ("Item-based CF", ItemBasedCFRecommender(k_neighbors=20), train),
        (
            "Matrix Factorization",
            MatrixFactorizationRecommender(
                n_factors=12,
                n_epochs=35,
                learning_rate=0.03,
                regularization=0.04,
                seed=42,
            ),
            train,
        ),
        (
            "Implicit BPR",
            ImplicitBPRRecommender(
                n_factors=12,
                n_epochs=35,
                learning_rate=0.05,
                regularization=0.01,
                seed=42,
            ),
            implicit_train,
        ),
    ]

    print("=== Evaluation @ 5 ===")
    result_rows = []
    fitted_models = {}

    for name, model, fit_data in models:
        model.fit(fit_data)
        fitted_models[name] = model
        metrics = evaluate_at_k(model, train=train, test=test, k=5, rating_threshold=4.0)
        result_rows.append({"model": name, **metrics})

    results = pd.DataFrame(result_rows)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()

    sample_user = train["user_id"].iloc[0]
    print(f"=== Sample recommendations for {sample_user} ===")

    for name, model in fitted_models.items():
        recommended_ids = model.recommend(sample_user, k=5, exclude_seen=True)
        titles = _titles_for_items(items, recommended_ids)
        print(f"{name:22s}: {', '.join(titles)}")


def _titles_for_items(items: pd.DataFrame, item_ids: list[str]) -> list[str]:
    title_by_id = dict(zip(items["item_id"], items["title"]))
    return [title_by_id.get(item_id, item_id) for item_id in item_ids]


if __name__ == "__main__":
    main()
