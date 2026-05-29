# 07. implicit feedback Answers

## 1

```python
from recsys_lab.data import generate_dataset, make_implicit_feedback

_, _, interactions = generate_dataset(seed=42)

for threshold in [3.0, 4.0, 5.0]:
    implicit = make_implicit_feedback(interactions, rating_threshold=threshold)
    print(threshold, implicit["preference"].sum())
```

## 2

`n_epochs` を増やすと positive item と sampled negative item の順位差をより学習します。ただし、増やしすぎると小さなデータでは過学習します。

## 3

implicit feedback には明確な負例がありません。未クリックは「嫌い」ではなく「知らない」かもしれないため、rating の誤差よりも上位に positive を出せるかを見る ranking metric が向いています。
