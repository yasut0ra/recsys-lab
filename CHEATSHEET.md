# Recommender Systems Cheatsheet

## データ表現

推薦の基本データは `user_id`, `item_id`, `rating`, `timestamp` の interaction log です。

```text
user_id  item_id  rating  timestamp
u001     i003     5       1
u001     i010     4       2
u002     i003     3       3
```

行列で見ると、ユーザー x アイテムの疎な評価行列 `R` になります。未観測は `0` や `NaN` として扱います。

## train/test split

推薦ではランダムに行を分けるだけだと、あるユーザーの履歴が test にしか存在しないことがあります。この教材ではユーザーごとに interaction を分け、各ユーザーの train 履歴を残すようにします。

## Precision@K

上位 `K` 件の推薦のうち、正解アイテムがどれだけ含まれたか。

```text
Precision@K = |recommended@K ∩ relevant| / K
```

推薦枠の「無駄の少なさ」を見る指標です。

## Recall@K

正解アイテムのうち、上位 `K` 件でどれだけ回収できたか。

```text
Recall@K = |recommended@K ∩ relevant| / |relevant|
```

取り逃しの少なさを見る指標です。

## NDCG@K

正解アイテムが上位に出ているほど高くなる指標です。

```text
DCG@K = sum(rel_i / log2(i + 1))
NDCG@K = DCG@K / ideal_DCG@K
```

順位の良さを見るときに便利です。

## 人気順推薦

最も単純な baseline です。アイテムごとの interaction 数や平均 rating で並べます。

強い点:

- 実装が簡単
- cold-start ユーザーにも出せる
- 比較対象として重要

弱い点:

- 個人化されない
- ニッチな嗜好を拾いにくい

## ユーザーベース協調フィルタリング

似ているユーザーを探し、そのユーザーたちが高く評価したアイテムを推薦します。

```text
score(u, i) = weighted average of ratings from similar users
```

ポイント:

- 類似度には cosine similarity を使うことが多い
- 評価の甘い/厳しいユーザー差を減らすため、ユーザー平均を引くとよい

## アイテムベース協調フィルタリング

ユーザーが過去に好んだアイテムと似ているアイテムを推薦します。

```text
score(u, i) = similarity(i, items_seen_by_u) weighted by user ratings
```

ユーザー数が多いサービスでは、ユーザー類似度よりアイテム類似度の方が安定しやすいことがあります。

## 行列分解

ユーザーとアイテムを低次元ベクトルで表し、内積で rating を予測します。

```text
rating_hat(u, i) = global_mean + user_bias[u] + item_bias[i] + dot(P[u], Q[i])
```

この教材では SGD で二乗誤差を小さくします。

## implicit feedback

閲覧、クリック、購入などは「好きではない」という明示的な負例がありません。そのため、観測された行動を positive、未観測を unknown として扱います。

代表的な考え方:

- rating を threshold で binary preference に変換する
- 未観測アイテムから negative sample を作る
- BPR のように「positive が negative より上に来る」よう学習する

## 実験の基本

1. baseline を作る
2. 同じ split と metric で比較する
3. `K`、近傍数、factor 数、epoch 数などを一つずつ変える
4. 指標だけでなく、実際の推薦リストも読む
