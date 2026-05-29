# 04. ユーザーベース協調フィルタリング Answers

## 1

`k_neighbors` が小さいと近いユーザーだけを使うため個人化は強くなりますが、ノイズに弱くなります。大きいと安定しますが、似ていないユーザーの影響も入ります。

## 2

平均を引かないと、常に高い rating を付けるユーザー同士が似ていると判定されやすくなります。嗜好ではなく rating の甘さを拾ってしまうことがあります。

## 3

```python
import numpy as np
from recsys_lab.data import generate_dataset
from recsys_lab.models import UserBasedCFRecommender

_, _, interactions = generate_dataset(seed=42)
model = UserBasedCFRecommender(k_neighbors=10).fit(interactions)

user_id = "u001"
user_pos = int(np.where(model.user_ids_ == user_id)[0][0])
sims = model.similarity_[user_pos].copy()
sims[user_pos] = 0.0

for pos in np.argsort(sims)[::-1][:3]:
    print(model.user_ids_[pos], sims[pos])
```
