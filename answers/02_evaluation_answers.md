# 02. train/test split と評価 Answers

## 1

`test_ratio` を大きくすると test interaction は増えますが、train に残る履歴が減ります。履歴が少ないモデルほど推薦が不安定になります。

## 2

`rating_threshold` を上げると relevant item が減ります。Relevant item が少ないほど Recall@K は変動しやすくなります。

## 3

Precision@K は推薦した K 件の中の当たり率、Recall@K は全 relevant item の回収率です。K を増やすと Recall は上がりやすい一方、Precision は下がりやすくなります。
