# 21. 総合演習：回答例

[問題へ戻る](../chapters/21_capstone.md)

## 1. validationで選んでからtestを見る

本文の候補設定をvalidationのNDCGで比較し、設定を固定します。trainとvalidationで再学習した後にtestを採点します。testで別の設定がよさそうに見えても、その結果を使って何度も選び直すのは避け、次の独立した評価計画にします。

## 2. 3つのseedでハイブリッドを比べる

```bash
.venv/bin/python -m recsys_lab.run_modern --seed 1 --output experiments/seed1.csv
.venv/bin/python -m recsys_lab.run_modern --seed 2 --output experiments/seed2.csv
.venv/bin/python -m recsys_lab.run_modern --seed 3 --output experiments/seed3.csv
```

各seed内でContent、EASE、Hybrid RRFを同じ条件で比べます。平均とばらつき、どのユーザーで変わったかを記録します。3回の練習データだけで実サービスの最良モデルを確定することはできません。

## 3. 次の実験の例

| 目的 | 方法の例 | 比較と評価 |
| --- | --- | --- |
| 新作品への対応 | 内容ベース、内容を使うTwo-Tower | trainに履歴のない作品を正解に持つ課題を別途作る |
| 精度 | ALS・EASE・ハイブリッドの設定比較 | validationで選び、testのRecallとNDCG |
| 多様性 | 同じ候補へのMMR | 精度と特徴ベースの多様性を併記 |
| 検索時間 | 厳密検索とANN | 候補品質、厳密検索の再現率、p95などの時間 |

生成器の最後の1作品はtestにも登場しません。したがってデモのスコアだけでは「cold-start作品への精度」を測れません。内容モデルがその作品の点数を計算できることと、実際に新作品を当てられることは区別します。
