"""Run reproducible validation/test experiments for the advanced curriculum."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from recsys_lab.data import generate_catalog_dataset
from recsys_lab.metrics import catalog_coverage, evaluate_at_k, intra_list_diversity
from recsys_lab.models import PopularityRecommender
from recsys_lab.modern_models import (
    ContentBasedRecommender,
    EASERecommender,
    HybridRRFRecommender,
    ImplicitALSRecommender,
    LightGCNRecommender,
    LinearTwoTowerRecommender,
    TransitionRecommender,
)
from recsys_lab.reranking import rerank_mmr
from recsys_lab.split import temporal_train_validation_test_split


class _MMRModel:
    def __init__(self, model, features, diversity_weight=0.3):
        self.model, self.features, self.diversity_weight = model, features, diversity_weight

    def recommend(self, user_id, k=5, exclude_seen=True):
        candidates = self.model.recommend(user_id, k=max(20, k), exclude_seen=exclude_seen)
        positions = [self.model.item_to_pos_[item] for item in candidates]
        scores = self.model.score_items(user_id)[positions]
        return rerank_mmr(candidates, scores, self.features[positions], k, self.diversity_weight)


def fit_models(train, items, seed, epochs):
    content = ContentBasedRecommender().fit(train, items)
    ease = EASERecommender().fit(train, item_ids=items["item_id"])
    tower = LinearTwoTowerRecommender(n_epochs=epochs, seed=seed).fit(train, items)
    return [
        ("Popularity", PopularityRecommender().fit(train)),
        ("Content TF-IDF", content),
        ("Implicit ALS", ImplicitALSRecommender(seed=seed).fit(train, item_ids=items["item_id"])),
        ("EASE", ease),
        ("Linear Two-Tower", tower),
        ("Transition", TransitionRecommender().fit(train, item_ids=items["item_id"])),
        ("LightGCN", LightGCNRecommender(n_epochs=epochs, seed=seed).fit(train, item_ids=items["item_id"])),
        ("Hybrid RRF", HybridRRFRecommender([content, ease])),
        ("Two-Tower + MMR", _MMRModel(tower, content.item_features_)),
    ], content


def run_experiment(seed=42, k=5, epochs=60):
    if k <= 0 or epochs <= 0:
        raise ValueError("k and epochs must be positive")
    users, items, interactions = generate_catalog_dataset(seed=seed)
    train, validation, test = temporal_train_validation_test_split(
        interactions, validation_size=2, test_size=2
    )
    print("=== Modern lab: synthetic content-aligned data ===")
    print(f"users/items: {len(users)} / {len(items)}")
    print(f"train/validation/test: {len(train)} / {len(validation)} / {len(test)}")
    print("Split: latest timestamp groups per user; all catalog metadata is known beforehand.")
    print("Preset settings; validation is reported before refitting train + validation for test.")
    rows = []
    final_models = None
    for phase, fit_data, held_out in [
        ("validation", train, validation),
        ("test", pd.concat([train, validation], ignore_index=True), test),
    ]:
        start = perf_counter()
        models, content = fit_models(fit_data, items, seed, epochs)
        print(f"\n=== {phase} @ {k} (all models fit in {perf_counter() - start:.2f}s) ===")
        eligible = held_out[held_out["rating"] >= 4]["user_id"].unique()
        phase_rows = []
        for name, model in models:
            metrics = evaluate_at_k(model, fit_data, held_out, k=k, rating_threshold=4)
            lists = [model.recommend(user, k=k) for user in eligible]
            diversity = []
            for recommended in lists:
                positions = [content.item_to_pos_[item] for item in recommended]
                diversity.append(intra_list_diversity(content.item_features_[positions]))
            row = {
                "phase": phase,
                "seed": seed,
                "k": k,
                "epochs": epochs,
                "model": name,
                **metrics,
                "coverage": catalog_coverage(lists, items["item_id"]),
                "diversity": float(np.mean(diversity)) if diversity else 0.0,
            }
            phase_rows.append(row)
        rows.extend(phase_rows)
        frame = pd.DataFrame(phase_rows)
        columns = [
            "model", "precision", "recall", "ndcg", "n_users", "coverage", "diversity"
        ]
        print(frame[columns].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        ))
        final_models = models
    sample_user = users.iloc[0]["user_id"]
    print(f"\n=== Final recommendations for {sample_user} ===")
    title_by_id = items.set_index("item_id")["title"].to_dict()
    for name, model in final_models:
        print(f"{name:20s}: {', '.join(title_by_id[item] for item in model.recommend(sample_user, k=k))}")
    print("\nCoverage/diversity describe exposure and text features, not user satisfaction.")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", type=Path, help="optional results CSV")
    args = parser.parse_args()
    if args.k < 1 or args.epochs < 1:
        parser.error("--k and --epochs must be positive")
    results = run_experiment(args.seed, args.k, args.epochs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
