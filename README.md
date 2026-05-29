# recsys-lab

Python だけで推薦システムの基礎を実装しながら学ぶための小さな教材リポジトリです。

外部ライブラリは最小限にし、`numpy` / `pandas` / `scikit-learn` だけを使います。実装はあえて読みやすさを優先し、推薦アルゴリズムの各ステップが追えるようにしています。

## できること

- 小さな擬似データセット生成
- 人気順推薦
- ユーザーベース協調フィルタリング
- アイテムベース協調フィルタリング
- 行列分解による推薦
- implicit feedback の扱い
- `Precision@K` / `Recall@K` / `NDCG@K` による評価
- ユーザー単位の train/test split
- 各章ごとの exercises と answers

## セットアップ

```bash
make init
```

## デモ実行

```bash
make run
```

`make run` は擬似データを生成し、同じ train/test split 上で複数の推薦手法を評価します。

## テスト

```bash
make check
```

## リポジトリ構成

```text
.
├── recsys_lab/          # 教材用の実装
├── chapters/            # 章ごとの解説と exercises
├── answers/             # 章ごとの回答例
├── tests/               # make check 用の軽量テスト
├── README.md
├── CHEATSHEET.md
├── Makefile
└── requirements.txt
```

## 学習順

1. [擬似データセット](chapters/01_dataset.md)
2. [train/test split と評価](chapters/02_evaluation.md)
3. [人気順推薦](chapters/03_popularity.md)
4. [ユーザーベース協調フィルタリング](chapters/04_user_based_cf.md)
5. [アイテムベース協調フィルタリング](chapters/05_item_based_cf.md)
6. [行列分解](chapters/06_matrix_factorization.md)
7. [implicit feedback](chapters/07_implicit_feedback.md)
8. [実験の進め方](chapters/08_experiment_workflow.md)

## 実装の入口

- データ生成: [recsys_lab/data.py](recsys_lab/data.py)
- 分割: [recsys_lab/split.py](recsys_lab/split.py)
- 評価指標: [recsys_lab/metrics.py](recsys_lab/metrics.py)
- 推薦モデル: [recsys_lab/models.py](recsys_lab/models.py)
- デモ実行: [recsys_lab/run.py](recsys_lab/run.py)
