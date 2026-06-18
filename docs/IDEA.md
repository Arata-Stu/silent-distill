# 実験定義書

## Silence- and Latency-Aware Self-Supervised Learning for Event Cameras

## 0. 本文書の目的

本書は、イベントカメラ向け自己教師あり学習手法の実験設計を定義するものである。

提案手法の目的は、以下の2点である。

1. 短いイベント蓄積時間でも、長い蓄積時間に近い有効な表現を得ること
2. イベントが少ない静かなシーンや低イベント密度シーンを、学習から捨てずに活用すること

本研究では、この目的に対して、短時間蓄積ビューと長時間蓄積ビューを用いた自己教師あり事前学習、および無イベント領域を利用した silence-aware objective を設計する。

---

# 1. 背景と問題意識

イベントカメラは、画素ごとの輝度変化がしきい値を超えたときに非同期イベントを出力する。イベントは通常、以下のように表される。

[
e_i = (x_i, y_i, t_i, p_i)
]

ここで、((x_i, y_i)) は空間座標、(t_i) はイベント時刻、(p_i) は極性である。

イベントカメラの利点は、高時間分解能、低レイテンシ、高ダイナミックレンジである。一方、実際の認識タスクでは、一定時間イベントを蓄積して voxel grid や event frame に変換してからニューラルネットワークに入力することが多い。

物体検出などでは、5 ms 程度の蓄積時間を用いた方が高精度になることが多い。しかし、長い蓄積時間を使うほどレイテンシは増加し、イベントカメラ本来の低遅延性は弱まる。

また、短い蓄積時間ではイベント数が少なくなるため、物体形状や運動の手がかりが不完全になりやすい。特に静かなシーンや低イベント密度シーンでは、既存の自己教師あり学習では学習信号が少なくなり、場合によっては前処理やサンプリングで除外されやすい。

しかし、イベントカメラにおいて「イベントが発生しないこと」は単なる欠損ではなく、輝度変化が発火しきい値を超えなかったというセンサー観測でもある。

本研究では、この性質を利用し、イベントが発生した領域だけでなく、イベントが発生しなかった時空間領域も学習信号として用いる。

---

# 2. 提案手法の概要

提案手法を仮に以下のように呼ぶ。

**SLA-SSL: Silence- and Latency-Aware Self-Supervised Learning for Event Cameras**

SLA-SSL は、以下の2つを中心とする自己教師あり事前学習である。

## 2.1 Short-to-Long Accumulation Distillation

同一時刻に対して、短時間蓄積イベントと長時間蓄積イベントを作る。

短時間蓄積イベント：

[
E_s(t) = E[t-\tau_s, t]
]

長時間蓄積イベント：

[
E_l(t) = E[t-\tau_l, t]
]

ただし、

[
\tau_s < \tau_l
]

である。

短時間蓄積ビューを student に入力し、長時間蓄積ビューを teacher に入力する。短時間ビューの特徴が長時間ビューの特徴に近づくように学習する。

狙いは、推論時には短時間窓しか使わずに、長時間窓に近い有効な表現を得ることである。

## 2.2 Silence-aware Learning

短時間窓内の各時空間ビンについて、イベントが発生したか、発生しなかったかを予測する。

イベントがあるビンを positive、イベントがないビンを negative として扱う。

これにより、イベント数が少ないシーンや静かなシーンでも、無イベント領域を自己教師信号として利用できる。

---

# 3. 本研究の主張

本研究の主張は以下である。

1. イベントカメラの自己教師あり学習において、蓄積時間の違いを view として利用できる。
2. 長時間蓄積ビューを教師、短時間蓄積ビューを生徒とすることで、短時間入力でも下流タスクに有効な表現を学習できる。
3. イベントが発生しなかった時空間領域を明示的な教師信号として扱うことで、静かなシーンや低イベント密度シーンも事前学習に利用できる。
4. 推論時には長時間蓄積や teacher network を使わず、短時間窓のみで低レイテンシな認識を行う。
5. 本手法は recurrent memory に依存せず、非recurrent backbone でも短時間窓の表現を改善することを目指す。

---

# 4. 入力定義

イベント列を以下のように定義する。

[
E = {(x_i, y_i, t_i, p_i)}_{i=1}^{N}
]

時刻 (t) に対して、短時間窓と長時間窓を作る。

[
E_s(t) = E[t-\tau_s, t]
]

[
E_l(t) = E[t-\tau_l, t]
]

基本設定では、以下を候補とする。

[
\tau_s \in {0.5, 1, 2}\mathrm{ms}
]

[
\tau_l \in {5, 10}\mathrm{ms}
]

最初の最小実験では、以下を用いる。

[
\tau_s = 1\mathrm{ms}
]

[
\tau_l = 5\mathrm{ms}
]

重要な点として、短時間窓と長時間窓はどちらも時刻 (t) までの過去イベントのみを使う。長時間窓が未来イベントを含まないようにする。

---

# 5. イベント表現

各イベント窓を voxel grid に変換する。

[
V_\tau(t) = \mathrm{Voxelize}(E[t-\tau,t])
]

voxel grid の形状は以下とする。

[
V_\tau(t) \in \mathbb{R}^{C \times B \times H \times W}
]

ここで、

* (C)：極性チャネル数
* (B)：時間ビン数
* (H, W)：空間解像度

である。

極性を考慮する場合、ONイベントとOFFイベントを別チャネルとして扱う。

極性を無視する ablation では、ON/OFF を統合して1チャネルとする。

時間軸の扱いについては、以下の表現を比較する。

1. **event frame**
   時間方向を潰して2D表現にする。

2. **temporal voxel**
   時間ビンを保持する。

3. **shuffled temporal voxel**
   時間ビンの順序をシャッフルする。

4. **fine temporal voxel**
   時間ビン数を増やす。

時間ビン数は、以下を候補とする。

[
B \in {1, 2, 5, 10}
]

---

# 6. ネットワーク構成

## 6.1 基本方針

本研究の主実験では、LSTM、GRU、RVT などの時系列モデルを前提にしない。

理由は、時系列モデルを使うと、短時間窓での性能改善が提案SSLによるものか、recurrent memory によるものかが分かりにくくなるためである。

したがって、主実験では非recurrent encoder を用いる。

## 6.2 Student encoder

短時間蓄積 voxel を student encoder に入力する。

[
F_s = f_\theta(V_s)
]

## 6.3 Teacher encoder

長時間蓄積 voxel を teacher encoder に入力する。

[
F_l = f_{\bar{\theta}}(V_l)
]

teacher encoder は student encoder の EMA によって更新する。

[
\bar{\theta} \leftarrow m\bar{\theta} + (1-m)\theta
]

## 6.4 Backbone 候補

主実験の候補は以下である。

* ResNet 系 CNN
* ConvNeXt 系 CNN
* Swin Transformer
* plain ViT
* イベント voxel 入力用の軽量 CNN backbone

RVT、GRU、LSTM は主実験ではなく、ablation で扱う。

---

# 7. 損失関数

最終的な損失は以下である。

[
\mathcal{L}
===========

\mathcal{L}*{S2L}
+
\lambda*{sil}\mathcal{L}*{silence}
+
\lambda*{pol}\mathcal{L}*{polarity}
+
\lambda*{rate}\mathcal{L}_{rate}
]

最小構成では、以下を用いる。

[
\mathcal{L}
===========

\mathcal{L}*{S2L}
+
\lambda*{sil}\mathcal{L}_{silence}
]

---

## 7.1 Short-to-Long Accumulation Distillation Loss

短時間窓の特徴を、長時間窓の特徴に近づける。

[
\mathcal{L}_{S2L}
=================

\left|
\mathrm{norm}(g(F_s))
---------------------

\mathrm{sg}(\mathrm{norm}(g(F_l)))
\right|_2^2
]

ここで、

* (F_s)：student feature
* (F_l)：teacher feature
* (g)：projection head
* (\mathrm{sg})：stop-gradient
* (\mathrm{norm})：特徴正規化

である。

multi-scale feature を用いる場合は、複数層で distillation を行う。

[
\mathcal{L}_{S2L}
=================

\sum_k
\left|
\mathrm{norm}(g_k(F_s^k))
-------------------------

\mathrm{sg}(\mathrm{norm}(g_k(F_l^k)))
\right|_2^2
]

---

## 7.2 Silence-aware Loss

短時間窓内の時空間ビンを以下のように表す。

[
b = (x, y, \tau, p)
]

イベントが存在するビンでは、

[
y_b = 1
]

イベントが存在しないビンでは、

[
y_b = 0
]

とする。

student feature から occupancy head により、各ビンにイベントが存在する確率を予測する。

[
q_b = h_{occ}(F_s)_b
]

損失は binary cross entropy とする。

[
\mathcal{L}_{silence}
=====================

-\sum_{b \in \mathcal{B}^{+}}
\log q_b
--------

\alpha
\sum_{b \in \mathcal{B}^{-}}
\log(1-q_b)
]

ここで、

* (\mathcal{B}^{+})：イベントが存在するビン
* (\mathcal{B}^{-})：イベントが存在しないビン
* (\alpha)：負例の重み

である。

---

## 7.3 Negative Sampling

無イベントビンは数が非常に多いため、全てを使わずサンプリングする。

### Random negative

イベントが存在しない時空間ビンをランダムに選ぶ。

目的は、静かなシーンや低イベント密度領域を学習に含めることである。

### Near-event negative

実イベントの近傍にあるが、イベントが存在しないビンを選ぶ。

目的は、物体境界、運動境界、イベント発生領域の周辺構造を学習することである。

### Hard negative

以下のような人工的な負例を作る。

* 実イベントの時刻を少しずらす
* 実イベントの空間位置を近傍ピクセルにずらす
* 実イベントの極性を反転する

目的は、ノイズイベントや誤った時空間対応に対して頑健な表現を学習することである。

---

## 7.4 Polarity Loss

イベントが存在するビンに対して、ON/OFF 極性を予測する。

[
\mathcal{L}_{polarity}
======================

-\sum_{b \in \mathcal{B}^{+}}
\sum_{p \in {+,-}}
p_b \log \hat{p}_b
]

この損失は補助損失として扱い、ablation によって有効性を検証する。

---

## 7.5 Event-rate Regularization

silence-aware loss だけでは、全てのビンを no-event と予測する方向に崩れる可能性がある。

そのため、予測イベント量が実際のイベント量に近づくように正則化する。

[
\mathcal{L}_{rate}
==================

\left|
\frac{1}{|\mathcal{B}|}\sum_b q_b
---------------------------------

\frac{|\mathcal{B}^{+}|}{|\mathcal{B}|}
\right|
]

この損失も補助損失として扱う。

---

# 8. 学習手順

事前学習はラベルなしイベント列を用いて行う。

1. イベントシーケンスから時刻 (t) をサンプリングする。
2. 短時間窓 (\tau_s) と長時間窓 (\tau_l) を決める。
3. 短時間イベント (E_s(t)) を取得する。
4. 長時間イベント (E_l(t)) を取得する。
5. それぞれを voxel grid に変換する。
6. student に短時間 voxel (V_s) を入力する。
7. teacher に長時間 voxel (V_l) を入力する。
8. short-to-long distillation loss を計算する。
9. 短時間 voxel 内の event / no-event ビンから silence-aware loss を計算する。
10. 必要に応じて polarity loss と event-rate regularization を計算する。
11. student encoder を更新する。
12. teacher encoder を EMA で更新する。

---

# 9. 推論時の設定

推論時には、teacher encoder と長時間窓を使用しない。

短時間窓のみを入力する。

[
\hat{y} = \mathrm{Head}(f_\theta(V_s))
]

つまり、推論時の条件は以下である。

* 入力：短時間窓
* teacher：なし
* long window：なし
* recurrent state：基本実験ではなし

本研究の狙いは、5 ms や 10 ms の蓄積に頼らず、0.5 ms、1 ms、2 ms といった短時間蓄積で性能低下を抑えることである。

---

# 10. 使用データセット

本研究では、単一のデータセットではなく、目的に応じて複数のデータセットを使い分ける。

データセットは以下の3層に分ける。

1. 主実験：高解像度イベント物体検出
2. 汎用表現評価：F³ と比較しやすい複数タスク・複数環境データセット
3. 小規模制御実験：背景イベントが比較的少ない非運転系データセット

---

## 10.1 主実験：Prophesee 1 Megapixel Automotive Detection Dataset

本研究の主実験では、Prophesee 1 Megapixel Automotive Detection Dataset を用いる。

本データセットは高解像度イベントカメラによる automotive detection dataset であり、車、歩行者、二輪車などの bounding box annotations を含む。

本研究では、以下の評価に用いる。

* 高解像度イベント物体検出
* 短時間蓄積入力での検出性能
* 5 ms 蓄積 baseline との性能差
* 0.5 / 1 / 2 / 5 ms における accumulation-time robustness curve
* event density subset による低イベント密度評価
* 背景イベントが多い実運用条件での有効性評価

Gen1 Automotive Detection Dataset は解像度が小さいため、本研究の主実験には用いない。

Gen1 を使う場合は、デバッグ、軽量な sanity check、既存実装の動作確認に限定する。

---

## 10.2 汎用表現評価：DSEC / MVSEC / M3ED

提案手法が物体検出専用ではなく、汎用的なイベント表現学習として有効かを確認するため、DSEC、MVSEC、M3ED を用いる。

これらは F³ でも利用されている系統のデータセットであり、予測型イベント表現との比較に有用である。

### DSEC

DSEC は driving scenario の stereo event dataset である。

本研究では、以下の評価に用いる。

* optical flow
* disparity / stereo
* semantic segmentation
* driving scene での dense prediction
* 昼夜・夕方・逆光など照明変化に対する頑健性
* 背景イベントが多い環境での評価

DSEC は運転シーンであるため背景イベントは多いが、dense prediction 評価に展開しやすい。

### MVSEC

MVSEC は、車両、バイク、ヘキサコプター、ハンドヘルドなど複数プラットフォームで収録された stereo event dataset である。

本研究では、以下の評価に用いる。

* optical flow
* depth
* pose / motion estimation
* driving 以外の platform を含む評価
* F³ との比較
* センサ・環境差に対する汎化評価

MVSEC は高解像度ではないが、タスクの幅があり、F³ 比較や低レベル視覚タスクの評価に有用である。

### M3ED

M3ED は、車両、四脚ロボット、ドローンなど複数プラットフォーム・複数環境で収録された高解像度イベントデータセットである。

本研究では、以下の評価に用いる。

* 高解像度イベント表現評価
* robotics scene での汎化
* drone / quadruped / vehicle のクロスプラットフォーム評価
* optical flow
* depth
* segmentation
* 高速運動
* off-road / forest / aggressive flight などの challenging scene
* 非運転シーンを含む汎用性評価

M3ED は、本研究の汎用表現評価において特に重要なデータセットとする。

---

## 10.3 小規模制御実験：非運転シーンデータセット

運転シーンでは背景イベントが多いため、silence-aware objective の効果を単純に切り分けることが難しい可能性がある。

そのため、小規模な制御実験として、背景が比較的単純な非運転系データセットも用いる。

候補は以下である。

* DVS Gesture
* N-Caltech101
* CIFAR10-DVS
* Poker-DVS
* N-Cars
* EVIMO2

これらは主実験ではなく、以下の目的で用いる。

* silence-aware loss の sanity check
* short-to-long distillation の基本効果確認
* 極性の有無の ablation
* 時間ビン数の ablation
* 背景イベントが少ない条件で no-event supervision が有効かの確認
* 小規模・短時間での実装検証

DVS Gesture は時間情報の有効性を見る小実験に向く。

N-Caltech101、CIFAR10-DVS、Poker-DVS は分類ベースの軽量実験に向く。

EVIMO2 は屋内シーン、物体運動、segmentation、depth などの制御実験に使える可能性がある。

---

## 10.4 データセットの位置づけ

| 位置づけ    | データセット                                               | 主な用途                                        |
| ------- | ---------------------------------------------------- | ------------------------------------------- |
| 主実験     | Prophesee 1 Mpx                                      | 高解像度イベント物体検出、短蓄積評価                          |
| 汎用表現評価  | M3ED                                                 | 高解像度、複数プラットフォーム、robotics scene、dense task   |
| 汎用表現評価  | DSEC                                                 | driving dense prediction、照明変化、stereo / flow |
| 汎用表現評価  | MVSEC                                                | flow / depth / pose、F³ 比較                   |
| 小規模制御実験 | DVS Gesture / N-Caltech101 / CIFAR10-DVS / EVIMO2 など | silence-aware の切り分け、軽量 ablation             |
| デバッグのみ  | Gen1                                                 | 軽量動作確認、既存実装確認                               |

---

# 11. 実験フェーズ

本研究は、段階的に実験を進める。

## 11.1 Phase 1：小規模制御実験

目的は、提案損失が基本的に機能するかを軽量に確認することである。

使用候補：

* DVS Gesture
* N-Caltech101
* CIFAR10-DVS
* Poker-DVS
* EVIMO2

確認する項目：

* (\mathcal{L}_{S2L}) が効くか
* (\mathcal{L}_{silence}) が効くか
* 極性を使うと改善するか
* 時間ビン数を増やすと改善するか
* 無イベント negative sampling が意味を持つか

この段階では、物体検出ではなく分類や簡易な downstream task でもよい。

## 11.2 Phase 2：主実験

目的は、高解像度イベント物体検出において、短蓄積での性能低下を抑えられるかを確認することである。

使用データセット：

* Prophesee 1 Megapixel Automotive Detection Dataset

確認する項目：

* 1 ms 入力での mAP 改善
* 2 ms 入力での mAP 改善
* 5 ms baseline との gap 縮小
* accumulation-time robustness curve
* event density subset での性能
* 背景イベントが多い運転シーンでの頑健性

## 11.3 Phase 3：汎用表現評価

目的は、提案手法が物体検出専用ではなく、汎用イベント表現として有効かを確認することである。

使用データセット：

* M3ED
* DSEC
* MVSEC

確認する項目：

* optical flow
* depth
* semantic segmentation
* cross-dataset transfer
* cross-platform transfer
* F³ との比較
* 低イベント密度条件での性能
* 高速運動条件での性能
* 昼夜・屋内外・off-road など環境差への頑健性

---

# 12. 下流タスク

## 12.1 主タスク：物体検出

主タスクは、高解像度イベント物体検出とする。

主に Prophesee 1 Mpx を用いる。

評価指標：

* mAP
* AP50
* AP75
* class-wise AP
* small / medium / large object AP
* accumulation time ごとの mAP
* latency と mAP のトレードオフ
* event density subset ごとの mAP

## 12.2 汎用表現タスク

M3ED、DSEC、MVSEC では以下を評価する。

* optical flow
* depth estimation
* semantic segmentation
* stereo / disparity
* pose / motion estimation

ただし、初期段階では全てを実施する必要はない。

まずは物体検出と optical flow / depth のいずれかに絞る。

## 12.3 小規模制御タスク

DVS Gesture、N-Caltech101、CIFAR10-DVS などでは、以下を行う。

* 分類精度
* short window accuracy
* event density ごとの分類性能
* ablation の高速検証

---

# 13. Baseline

比較対象は以下とする。

## 13.1 Supervised from scratch

ラベル付きデータのみで学習する。

蓄積時間を変えて比較する。

* 0.5 ms
* 1 ms
* 2 ms
* 5 ms
* 10 ms

## 13.2 既存SSL

以下のような既存の自己教師あり学習手法と比較する。

* MAE / MEM-style masked event modeling
* DINO-style feature matching
* JEPA-style latent prediction
* future event prediction
* F³

## 13.3 Long-window baseline

長時間蓄積を用いる通常の高精度 baseline として、5 ms または 10 ms 入力を用いる。

本研究の目標は、1 ms または 2 ms 入力で、この long-window baseline との性能差を縮めることである。

## 13.4 Recurrent model baseline

RVT、GRU、LSTM などの時系列モデルを用いる。

ただし、これは主比較ではなく、補助比較とする。

時系列モデルを使った結果は、recurrent memory の効果と提案SSLの効果を分けて解釈する必要がある。

---

# 14. Main Experiments

## 14.1 実験1：短時間蓄積での物体検出性能

目的は、Prophesee 1 Mpx において、提案手法が短時間窓での物体検出性能を改善するかを確認することである。

比較条件：

| 方法                              |  入力窓 | 事前学習 | recurrent |
| ------------------------------- | ---: | ---- | --------- |
| Scratch                         | 1 ms | なし   | なし        |
| Scratch                         | 5 ms | なし   | なし        |
| MEM-style SSL                   | 1 ms | あり   | なし        |
| DINO-style SSL                  | 1 ms | あり   | なし        |
| F³-style predictive pretraining | 1 ms | あり   | なし        |
| SLA-SSL                         | 1 ms | あり   | なし        |

期待される結果は、SLA-SSL が 1 ms 入力において scratch や既存SSLより高い mAP を示し、5 ms baseline との性能差を縮小することである。

---

## 14.2 実験2：蓄積時間に対する頑健性

目的は、蓄積時間を短くしたときの性能低下を、提案手法が抑えられるかを確認することである。

評価する入力窓：

[
\tau_s \in {0.5, 1, 2, 5, 10}\mathrm{ms}
]

各蓄積時間で mAP を評価し、accumulation-time robustness curve を作成する。

期待される結果は、短時間側、特に 0.5 ms、1 ms、2 ms において、提案手法の性能低下が小さいことである。

---

## 14.3 実験3：低イベント密度シーンでの評価

目的は、静かなシーンやイベント数が少ないシーンに対して、silence-aware learning が有効かを確認することである。

各評価サンプルを event density に基づいて分割する。

例：

* low event density
* medium event density
* high event density

それぞれの subset で mAP または task metric を計算する。

期待される結果は、low event density subset において SLA-SSL が baseline より高い性能を示すことである。

---

## 14.4 実験4：非運転データセットでの制御実験

目的は、運転シーン特有の背景イベントの影響を除き、silence-aware objective の基本効果を確認することである。

使用候補：

* DVS Gesture
* N-Caltech101
* CIFAR10-DVS
* EVIMO2

評価項目：

* short-to-long distillation の有無
* silence-aware loss の有無
* 極性の有無
* 時間ビン数
* event density subset

期待される結果は、背景イベントが少ない条件でも、無イベント領域を利用することで表現が改善することである。

---

## 14.5 実験5：F³ 系データセットでの汎用表現評価

目的は、提案手法が物体検出に限らず、汎用イベント表現として有効かを確認することである。

使用データセット：

* DSEC
* MVSEC
* M3ED

評価タスク：

* optical flow
* depth
* segmentation
* stereo / disparity

比較対象：

* scratch
* MEM / MAE-style pretraining
* future event prediction
* F³
* SLA-SSL

期待される結果は、SLA-SSL が特に短時間窓、低イベント密度条件、event-rate variation のある条件で有効性を示すことである。

---

# 15. Ablation Study

## 15.1 Loss component ablation

各損失の寄与を確認する。

| 条件 | (\mathcal{L}_{S2L}) | (\mathcal{L}_{silence}) | (\mathcal{L}_{polarity}) | (\mathcal{L}_{rate}) |
| -- | ------------------- | ----------------------- | ------------------------ | -------------------- |
| A  | あり                  | なし                      | なし                       | なし                   |
| B  | なし                  | あり                      | なし                       | なし                   |
| C  | あり                  | あり                      | なし                       | なし                   |
| D  | あり                  | あり                      | あり                       | なし                   |
| E  | あり                  | あり                      | あり                       | あり                   |

目的は、short-to-long distillation と silence-aware loss のどちらが効いているか、また両者の組み合わせが有効かを確認することである。

---

## 15.2 時系列モデルの利用

時系列モデルを使うかどうかを検証する。

| 条件 | backbone                   | recurrent |
| -- | -------------------------- | --------- |
| A  | CNN / ViT                  | なし        |
| B  | CNN / ViT + SLA-SSL        | なし        |
| C  | GRU / LSTM / RVT           | あり        |
| D  | GRU / LSTM / RVT + SLA-SSL | あり        |

主実験では非recurrent backbone を用いる。

時系列モデルの実験は、提案手法が recurrent model と併用可能かを確認する補助実験とする。

---

## 15.3 蓄積時間

student window と teacher window の組み合わせを変える。

| student window | teacher window | 目的                   |
| -------------: | -------------: | -------------------- |
|         0.5 ms |           5 ms | 極短時間入力での効果           |
|           1 ms |           5 ms | 基本設定                 |
|           2 ms |           5 ms | やや情報量がある短時間入力        |
|           1 ms |          10 ms | teacher を長くした場合      |
|           1 ms |          20 ms | 長すぎる teacher の影響     |
|           5 ms |           5 ms | short-to-long 性がない場合 |

目的は、最適な teacher window と student window の関係を検証することである。

---

## 15.4 極性の扱い

極性を使うかどうかを検証する。

| 条件 | 入力極性 | polarity loss | hard negative 極性反転 |
| -- | ---- | ------------- | ------------------ |
| A  | なし   | なし            | なし                 |
| B  | あり   | なし            | なし                 |
| C  | あり   | あり            | なし                 |
| D  | あり   | あり            | あり                 |

目的は、ON/OFF 極性が表現学習にどの程度寄与するかを確認することである。

---

## 15.5 時間軸表現

時間情報の扱いを検証する。

| 条件 | 表現                      | 内容              |
| -- | ----------------------- | --------------- |
| A  | event frame             | 時間方向に全て足し合わせる   |
| B  | temporal voxel          | 時間ビンを保持する       |
| C  | shuffled temporal voxel | 時間ビンの順序をシャッフルする |
| D  | fine temporal voxel     | 時間ビン数を増やす       |

時間ビン数 (B) については以下を検証する。

[
B \in {1, 2, 5, 10}
]

目的は、短時間蓄積において時間順序や時間分解能がどれほど重要かを確認することである。

---

## 15.6 Negative sampling

Silence-aware loss における負例の取り方を検証する。

| 条件 | negative sampling                   |
| -- | ----------------------------------- |
| A  | random negative のみ                  |
| B  | near-event negative のみ              |
| C  | hard negative のみ                    |
| D  | random + near-event                 |
| E  | random + near-event + hard negative |

目的は、無イベント領域を単純な負例として使うだけで十分か、イベント近傍や hard negative が必要かを確認することである。

---

## 15.7 データセット依存性

データセットごとに提案手法の効果がどう変わるかを確認する。

| データセット群                      | 期待される観察                      |
| ---------------------------- | ---------------------------- |
| Prophesee 1 Mpx              | 高解像度・背景イベント多めの物体検出で効くか       |
| DSEC                         | driving dense task と照明変化で効くか |
| MVSEC                        | flow / depth / pose 系で効くか    |
| M3ED                         | 高解像度・robotics・非運転環境でも効くか     |
| DVS Gesture / N-Caltech101 等 | 背景が比較的単純な条件で基本効果が見えるか        |

---

# 16. 評価指標

## 16.1 物体検出

* mAP
* AP50
* AP75
* class-wise AP
* small / medium / large object AP
* accumulation time ごとの mAP
* event density ごとの mAP
* latency と mAP のトレードオフ

## 16.2 Optical flow

* Average Endpoint Error
* percentage of outliers
* latency ごとの flow accuracy
* event density ごとの flow accuracy

## 16.3 Depth

* Abs Rel
* RMSE
* (\delta < 1.25)
* depth error by event density

## 16.4 Semantic segmentation

* mIoU
* class-wise IoU
* event density subset ごとの mIoU

## 16.5 分類

* top-1 accuracy
* top-5 accuracy
* short-window accuracy
* event density subset ごとの accuracy

---

# 17. 最小実装案

最初に実装する最小構成は以下である。

## 17.1 Pretraining

* dataset：小規模制御データセット、または Prophesee 1 Mpx の unlabeled training split
* input：event voxel
* student window：1 ms
* teacher window：5 ms
* backbone：非recurrent CNN または ViT
* teacher：EMA teacher
* loss：(\mathcal{L}*{S2L} + \lambda*{sil}\mathcal{L}_{silence})
* negative sampling：random + near-event

## 17.2 Fine-tuning

* 主タスク：物体検出
* 主データセット：Prophesee 1 Mpx
* 入力：1 ms event voxel
* teacher：使用しない
* long window：使用しない
* recurrent：基本実験では使用しない

## 17.3 Main comparison

* scratch 1 ms
* scratch 5 ms
* existing SSL 1 ms
* SLA-SSL 1 ms

まずはこの最小構成で、1 ms 入力時の性能改善と、5 ms baseline との gap 縮小を確認する。

---

# 18. 期待される結果

本手法で期待する結果は以下である。

1. 1 ms または 2 ms 入力において、scratch や既存SSLより高い性能を示す。
2. 5 ms 入力 baseline との性能差を縮小する。
3. low event density subset において、silence-aware loss の効果が大きくなる。
4. temporal voxel 表現は event frame より有効である。
5. 極性情報を考慮することで性能が改善する可能性がある。
6. 非recurrent backbone でも効果が確認できる。
7. recurrent model と併用した場合には、追加の性能向上が期待できる。
8. Prophesee 1 Mpx のような背景イベントが多い運転環境だけでなく、M3ED や非運転系データセットでも一定の効果が確認できる可能性がある。

---

# 19. 注意点とリスク

## 19.1 短時間窓から長時間窓の情報を完全に復元するわけではない

本手法は、短時間窓から長時間窓の観測そのものを復元することを目的としない。

短時間窓に物体の手がかりが全く含まれていない場合、長時間窓と同等の情報を得ることは原理的に難しい。

目的は、短時間窓でも下流タスクに有効な表現を学習することである。

## 19.2 運転シーンでは背景イベントが多い

Prophesee 1 Mpx や DSEC のような運転シーンでは、背景イベントが多く発生する。

そのため、silence-aware loss の効果を単純に解釈するのが難しい可能性がある。

この問題に対して、小規模な非運転データセットや M3ED / EVIMO2 などを用いた補助実験で、背景イベントが少ない条件や制御された条件での効果を確認する。

## 19.3 時系列モデルの効果と提案SSLの効果を分離する必要がある

RVT、LSTM、GRU などを使うと、性能向上が recurrent memory によるものか、提案SSLによるものか分かりにくくなる。

そのため、主実験では非recurrent backbone を用いる。

時系列モデルは、提案手法との併用可能性を見るための補助実験とする。

---

# 20. 最終要約

本研究では、イベントカメラ向けの自己教師あり事前学習手法として、SLA-SSL を提案する。

SLA-SSL は、同一時刻に対して短時間蓄積ビューと長時間蓄積ビューを作成し、短時間ビューの特徴が長時間ビューの特徴に近づくように学習する。さらに、イベントが発生しなかった時空間領域を明示的な負例として利用することで、静かなシーンや低イベント密度シーンも学習に含める。

主実験では、Gen1 ではなく Prophesee 1 Megapixel Automotive Detection Dataset を用い、高解像度イベント物体検出における短蓄積性能を評価する。

汎用表現評価では、DSEC、MVSEC、M3ED を用い、F³ と比較しやすい optical flow、depth、segmentation などのタスクを検討する。

さらに、運転シーンでは背景イベントが多いため、小規模な非運転データセットを用いた制御実験も行う。

本手法の最終的な目標は、イベントカメラ本来の低レイテンシ性を維持しながら、短時間蓄積・低イベント密度条件でも頑健な表現を学習することである。
