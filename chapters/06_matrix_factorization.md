# 06. 行列分解

行列分解は、ユーザーとアイテムを低次元ベクトルで表し、内積で rating を予測します。

見るファイル:

- `recsys_lab/models.py` の `MatrixFactorizationRecommender`

重要な考え方:

- `user_factors` と `item_factors` を学習する
- global mean, user bias, item bias を足すと安定しやすい
- この教材では SGD で二乗誤差を小さくする

## Exercises

1. `n_factors` を `4`, `12`, `32` に変えて評価してください。
2. `n_epochs` を増やすと何が起きるか確認してください。
3. `regularization=0.0` にしたとき、過学習しやすくなる理由を説明してください。

## Answers

回答例は `answers/06_matrix_factorization_answers.md` にあります。
