# 07. implicit feedback

implicit feedback は、クリック、閲覧、購入などの行動ログです。明示的な低評価がないため、未観測を「嫌い」とはみなしません。

見るファイル:

- `recsys_lab/data.py` の `make_implicit_feedback`
- `recsys_lab/models.py` の `ImplicitBPRRecommender`

重要な考え方:

- 高 rating を positive preference に変換する
- 未観測アイテムは true negative ではなく unknown と考える
- BPR は positive item が sampled negative item より上に来るように学習する

## Exercises

1. `rating_threshold` を変えて positive 数がどう変わるか確認してください。
2. `ImplicitBPRRecommender` の `n_epochs` を変えて評価してください。
3. implicit feedback で accuracy や RMSE より ranking metric が使いやすい理由を説明してください。

## Answers

回答例は `answers/07_implicit_feedback_answers.md` にあります。
