# 05. アイテムベース協調フィルタリング Answers

## 1

近傍数が小さいと似ているアイテムだけに寄ります。近傍数が大きいと多くの評価を使えますが、弱い類似も混ざります。

## 2

```python
import numpy as np
from recsys_lab.data import generate_dataset
from recsys_lab.models import ItemBasedCFRecommender

_, _, interactions = generate_dataset(seed=42)
model = ItemBasedCFRecommender(k_neighbors=20).fit(interactions)

item_id = "i001"
item_pos = int(np.where(model.item_ids_ == item_id)[0][0])
sims = model.similarity_[item_pos].copy()
sims[item_pos] = 0.0

for pos in np.argsort(sims)[::-1][:5]:
    print(model.item_ids_[pos], sims[pos])
```

## 3

アイテム平均を引かないと、人気アイテムや高評価されやすいアイテム同士が似ていると判定されやすくなります。
