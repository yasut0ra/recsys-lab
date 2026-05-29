# 03. 人気順推薦 Answers

## 1

平均 rating だけで並べると、たまたま 1 件だけ 5 点だったアイテムが上位に来ることがあります。interaction 数を併用すると、信頼できる人気を表しやすくなります。

## 2

```python
from recsys_lab.data import generate_dataset
from recsys_lab.models import PopularityRecommender

_, _, interactions = generate_dataset(seed=42)
model = PopularityRecommender().fit(interactions)

print(model.recommend("u001", k=5, exclude_seen=False))
```

## 3

```python
print(model.recommend("u999", k=5))
```

未知ユーザーには個人履歴がないため、人気順推薦は自然な fallback になります。
