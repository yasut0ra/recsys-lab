# 02. train/test split と評価

推薦では「あるユーザーに対して、未知のアイテムをどれだけ当てられるか」を評価します。

見るファイル:

- `recsys_lab/split.py`
- `recsys_lab/metrics.py`

重要な考え方:

- ユーザーごとに train/test を分ける
- test の高評価アイテムを relevant item として扱う
- 上位 K 件の品質を `Precision@K`, `Recall@K`, `NDCG@K` で見る

## Exercises

1. `test_ratio` を `0.1`, `0.3`, `0.5` に変えて、評価対象ユーザー数を比べてください。
2. `rating_threshold` を `3.0`, `4.0`, `5.0` に変えて、指標がどう変わるか確認してください。
3. `Precision@K` と `Recall@K` がトレードオフになりやすい理由を説明してください。

## Answers

回答例は `answers/02_evaluation_answers.md` にあります。
