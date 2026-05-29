# 04. ユーザーベース協調フィルタリング

ユーザーベース協調フィルタリングは「自分と似ているユーザーが好んだアイテム」を推薦します。

見るファイル:

- `recsys_lab/models.py` の `UserBasedCFRecommender`

重要な考え方:

- user-item matrix を作る
- ユーザーごとの平均 rating を引いて rating 傾向を補正する
- cosine similarity で似ているユーザーを探す
- 近傍ユーザーの rating を類似度で重み付けする

## Exercises

1. `k_neighbors` を `3`, `10`, `30` に変えて評価してください。
2. ユーザー平均を引かない実装に変えると、どのような問題が起きるか説明してください。
3. 任意のユーザーについて、最も似ているユーザー上位 3 人を表示してください。

## Answers

回答例は `answers/04_user_based_cf_answers.md` にあります。
