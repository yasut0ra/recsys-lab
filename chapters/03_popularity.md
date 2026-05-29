# 03. 人気順推薦

人気順推薦は、最も重要な baseline です。個人化はしませんが、実サービスでも強いことがあります。

見るファイル:

- `recsys_lab/models.py` の `PopularityRecommender`

重要な考え方:

- まずは interaction 数でアイテムを並べる
- 同点のときは平均 rating などで tie-break する
- 学習済みユーザーには既視聴アイテムを除外して推薦する

## Exercises

1. 人気順を interaction 数ではなく平均 rating 順に変更したら何が起きるか考えてください。
2. `exclude_seen=False` で推薦し、既に見たアイテムが混ざることを確認してください。
3. cold-start ユーザー `u999` に推薦を出してください。

## Answers

回答例は `answers/03_popularity_answers.md` にあります。
