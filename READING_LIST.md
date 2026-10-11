# 原論文と公式資料の案内

[教材の入口](README.md) / [9章：手法の地図](chapters/09_modern_landscape.md) / [21章：総合演習](chapters/21_capstone.md)

本文の図と小さな実装で仕組みをつかんだ後に、原論文の図、入力、学習目標、評価条件を読んでください。最初から全ての式を追う必要はありません。

以下は2026年10月12日に確認した一次資料です。網羅とは、主要な設計の系統を学べる範囲を指します。すべての派生手法や、論文で報告された実験の再実装を含むものではありません。年は主な公表年で、arXivへの初回投稿と会議での発表が違う場合は併記しています。

## 候補検索、内容、比較の基準

| 資料 | どこを読む？ | 対応する章 |
| --- | --- | --- |
| [Deep Neural Networks for YouTube Recommendations（2016）](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) | 候補生成とランキングを分ける全体図。論文中のモデルと本教材の線形Two-Towerは同一ではありません | [9章](chapters/09_modern_landscape.md)、[12章](chapters/12_two_tower_retrieval.md) |
| [scikit-learn：TF-IDFの公式説明](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) | TF、IDF、正規化と実装の定義 | [10章](chapters/10_content_hybrid.md) |
| [Reciprocal Rank Fusion（2009、著者公開PDF）](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | 点数の尺度をそろえずに順位を混ぜる式。情報検索の手法を推薦の融合に使っています | [10章](chapters/10_content_hybrid.md) |
| [Collaborative Filtering for Implicit Feedback Datasets（2008、著者公開PDF）](https://yifanhu.net/PUB/cf.pdf) | 好みと確信度を分ける重み付き目的関数、交互最小二乗 | [11章](chapters/11_strong_baselines.md) |
| [Embarrassingly Shallow Autoencoders for Sparse Data / EASE（2019）](https://arxiv.org/abs/1905.03375) | 対角を0にする制約と閉形式の解 | [11章](chapters/11_strong_baselines.md) |
| [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations（2019）](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) | 学習時の候補のサンプリング確率と補正 | [12章](chapters/12_two_tower_retrieval.md) |
| [Faiss公式Wiki](https://github.com/facebookresearch/faiss/wiki) | ベクトル検索、インデックスの選択、速度と検索精度の釣り合い | [12章](chapters/12_two_tower_retrieval.md) |

## 特徴と履歴を使うランキング

| 原論文 | どこを読む？ | 対応する章 |
| --- | --- | --- |
| [Factorization Machines / FM（2010）](https://ieeexplore.ieee.org/document/5694074) | 疎な特徴同士の交差を、低次元の内積で表す式 | [13章](chapters/13_feature_ranking.md) |
| [DeepFM（2017）](https://arxiv.org/abs/1703.04247) | FMと深いネットワークが埋め込みを共有する図 | [13章](chapters/13_feature_ranking.md) |
| [DCN V2（2020プレプリント、2021会議）](https://arxiv.org/abs/2008.13535) | 明示的な特徴交差と、モデルの配置の違い | [13章](chapters/13_feature_ranking.md) |
| [Deep Interest Network / DIN（2017プレプリント、2018会議）](https://arxiv.org/abs/1706.06978) | 候補ごとに履歴の参考の仕方を変える図 | [13章](chapters/13_feature_ranking.md) |
| [Multi-gate Mixture-of-Experts / MMoE（2018）](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) | 共有expertとタスク別gate。クリックと購入など、複数目標の関係 | [13章](chapters/13_feature_ranking.md) |
| [Self-Attentive Sequential Recommendation / SASRec（2018）](https://arxiv.org/abs/1808.09781) | 時系列の自己注意、未来を隠す制約、次の作品を学ぶ目標 | [14章](chapters/14_sequential_recommendation.md) |
| [BERT4Rec（2019）](https://arxiv.org/abs/1904.06690) | 観測済みの履歴の一部を隠して復元する学習。未来のtestを見てよいという意味ではありません | [14章](chapters/14_sequential_recommendation.md) |

## グラフ、復元、生成

| 原論文 | どこを読む？ | 対応する章 |
| --- | --- | --- |
| [LightGCN（2020）](https://arxiv.org/abs/2002.02126) | 正規化した近傍集約と、層ごとの埋め込みを合わせる式 | [15章](chapters/15_graph_recommendation.md) |
| [Self-supervised Graph Learning for Recommendation / SGL（2020プレプリント、2021会議）](https://arxiv.org/abs/2010.10783) | グラフの異なる見方を作る工夫と、対照学習の目標 | [15章](chapters/15_graph_recommendation.md) |
| [How Powerful is Graph Convolution for Recommendation? / SimGCL（2021プレプリント、2022会議）](https://arxiv.org/abs/2108.07567) | 埋め込みへのノイズで見方を変える実験 | [15章](chapters/15_graph_recommendation.md) |
| [Variational Autoencoders for Collaborative Filtering / Mult-VAE（2018）](https://arxiv.org/abs/1802.05814) | 多項分布を使う復元の目標と、VAEの正則化 | [16章](chapters/16_vae_diffusion.md) |
| [Diffusion Recommender Model / DiffRec（2023）](https://arxiv.org/abs/2304.04971) | 行動ベクトルへのノイズ付加と、学習した除去過程 | [16章](chapters/16_vae_diffusion.md) |
| [Recommender Systems with Generative Retrieval / TIGER（2023）](https://arxiv.org/abs/2305.05065) | Semantic IDの作成、符号の生成、作品IDへの復号 | [17章](chapters/17_generative_recommendation.md) |
| [Actions Speak Louder than Words / HSTU（2024）](https://arxiv.org/abs/2402.17152) | 大規模な行動列の定式化とHierarchical Sequential Transduction Unit。TIGERの符号生成とは設計が違います | [17章](chapters/17_generative_recommendation.md) |

## 言葉、画像、多様性、評価

| 原論文 | どこを読む？ | 対応する章 |
| --- | --- | --- |
| [Recommendation as Language Processing / P5（2022）](https://arxiv.org/abs/2203.13366) | 複数の推薦タスクを言葉の入出力にそろえる設計 | [18章](chapters/18_llm_multimodal.md) |
| [Learning Transferable Visual Models From Natural Language Supervision / CLIP（2021）](https://arxiv.org/abs/2103.00020) | 画像と言葉を対応させる学習。推薦専用のモデルではなく、特徴として利用する出発点です | [18章](chapters/18_llm_multimodal.md) |
| [The Use of MMR, Diversity-Based Reranking…（1998、著者公開PDF）](https://www.cs.cmu.edu/~jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf) | 相性と、選択済みの対象との重複を合わせる再ランキング | [19章](chapters/19_reranking_bandits.md) |
| [A Contextual-Bandit Approach to Personalized News Article Recommendation / LinUCB（2010）](https://arxiv.org/abs/1003.0146) | 文脈に応じた報酬の予測と不確実性。本文のUCB計算は文脈を持たない導入例です | [19章](chapters/19_reranking_bandits.md) |
| [Counterfactual Risk Minimization（2015）](https://arxiv.org/abs/1502.02362) | ログの露出確率、反実仮想の推定、分散への対処 | [20章](chapters/20_evaluation_serving.md) |

## 2026年の研究を追う入口

[RecSys 2026の公式セッション4](https://recsys.acm.org/recsys26/session-4/)では、系列と生成型、エージェントを使う推薦の研究を確認できます。例えばRecRecは、履歴を潜在的な興味へ圧縮し、再帰的な推論で次の候補を扱う研究です。モデル選択にエージェントを使う研究も掲載されています。

ここは本教材の実装済みモデルとは別の発展的な読み物です。公開された要約や名称だけで、学習目標や運用の有効性を断定せず、原論文の実験条件へ進んでください。

## 公開データと大きな実験へ進む

- [MovieLens公式](https://grouplens.org/datasets/movielens/)：データの版、利用条件、安定したベンチマークかを確認する入口。[100Kの案内](https://grouplens.org/datasets/movielens/100k/)から、小さな固定データへ進めます。
- [RecBoleの原論文（2020）](https://arxiv.org/abs/2011.01731)と[公式ドキュメント](https://recbole.io/docs/)：複数の推薦手法を共通の枠組みで実験する入口。ライブラリの既定値に任せず、分割・負例・候補集合・指標を記録します。
- [Faiss公式Wiki](https://github.com/facebookresearch/faiss/wiki)：実際にANNを導入する際の検索・インデックスの資料。本教材のデモは全件を比較する正確な検索です。

本格的な実装へ進むときは、各論文が紹介するコードのリンクから、著者や研究組織が公開したリポジトリを確認してください。データ条件や学習方法が変わるため、本教材の結果と論文の数値をそのまま比較しません。

## 論文を読むときの5つの問い

1. **入力は何か。** IDだけか、行動の順番、説明、画像、表示記録もあるか。
2. **何を学ぶか。** 点数、順番、次の作品、分布、符号列のどれか。
3. **何と比べたか。** 人気順や強い単純な基準も、同じ条件で評価したか。
4. **どの情報を隠したか。** 時間、候補集合、露出の偏りはどう扱ったか。
5. **何が増えたか。** 精度だけでなく、計算量、メモリー、学習時間、配信の仕事を確認する。
