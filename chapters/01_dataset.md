# 01. 擬似データセット

この章では、推薦システムの実験で使う小さな interaction log を作ります。

見るファイル:

- `recsys_lab/data.py`

重要な考え方:

- 推薦データは `user_id`, `item_id`, `rating`, `timestamp` の行ログとして持てる
- 行ログは user-item matrix に変換できる
- 擬似データでも、ユーザー嗜好、アイテム人気、ノイズを入れると実験しやすい

## Exercises

1. `generate_dataset(seed=1)` と `generate_dataset(seed=2)` の結果を比べてください。
2. `min_interactions_per_user` と `max_interactions_per_user` を変えて、評価結果がどう変わるか確認してください。
3. `build_user_item_matrix` を使い、行数、列数、非ゼロ要素数を出してください。

## Answers

回答例は `answers/01_dataset_answers.md` にあります。
