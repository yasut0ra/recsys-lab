# 01. 擬似データセット Answers

## 1

`seed` を変えると、ユーザー嗜好、アイテム特徴量、interaction のサンプリング結果が変わります。同じ `seed` なら毎回同じデータになるため、実験比較がしやすくなります。

## 2

interaction 数を増やすと、協調フィルタリングや行列分解が学習に使える情報が増えます。一方で、test にも情報が増えるため、評価対象の relevant item 数も変わります。

## 3

```python
from recsys_lab.data import generate_dataset, build_user_item_matrix

_, _, interactions = generate_dataset(seed=42)
matrix = build_user_item_matrix(interactions)

print(matrix.shape)
print((matrix.to_numpy() > 0).sum())
```
