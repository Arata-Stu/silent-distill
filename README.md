# SLA-SSL

イベントカメラ向け **Silence- and Latency-Aware Self-Supervised Learning** の研究実装です。
同じ終端時刻 `t` を持つ短時間窓を student、長時間窓を EMA teacher に入力し、短時間表現を
長時間表現へ distill します。同時に、短時間窓の event / no-event occupancy を予測します。

このリポジトリは実験定義書の最小実装を対象にしています。

- SLA-SSL pretraining: S2L、silence、polarity、event-rate loss
- random / near-event / shifted hard negative sampling
- 非 recurrent ResNet-18 / ResNet-50 encoder
- 小規模分類 fine-tuning
- Prophesee 1 Mpx の FCOS 物体検出 fine-tuning
- DSEC / M3ED / MVSEC native HDF5からのSSL pretraining
- accumulation time、polarity、temporal bins、negative sampling の ablation
- event-density 別 accuracy / COCO mAP
- single GPU、`torchrun` DDP、AMP、resume、Slurm
- Hydraによる日時・run name別の実験管理
- TensorBoardへのtrain/validation metric記録
- sequence loader、GRU/LSTM、plain ViT、multi-scale distillation

すべての event window は `[t - tau, t]` から作り、future event は使用しません。推論時は
student と short window だけを使用します。

## 実行環境

この Mac での学習実行は想定していません。推奨環境は Linux、NVIDIA GPU、CUDA 12 系、
Python 3.10 以上です。Metavision SDKは使用しません。HDF5は`h5py`で直接読みます。
DSECのBlosc圧縮filterはPyPIの`hdf5plugin`が登録します。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[eval,dev]"
```

CUDA に合う PyTorch wheel が必要な場合は、先に PyTorch 公式手順で `torch` / `torchvision`
を導入してください。Docker を使う場合は次の通りです。

```bash
docker build -t sla-ssl .
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD:/workspace/sla-ssl" \
  -v /path/to/datasets:/datasets sla-ssl
```

## データ準備

データセット本体はライセンス上このリポジトリに含めません。1 Mpxは既存のevent H5と
`*_bbox.npy`をそのまま使用します。H5をコピー・変換せず、manifestだけを作成します。

```bash
for SPLIT in train val test; do
  sla-index-h5 --dataset 1mpx \
    --data-root /datasets/prophesee_1mpx \
    --output-dir /datasets/prophesee_1mpx/manifests \
    --split "$SPLIT" --class-ids 0,1,2 \
    --long-window-us 50000
done
```

H5はsplitされた`x/y/t/p`、compound `x/y/t/p`、または`[x,y,t,p]` matrixを自動判定します。
bboxのraw class `0,1,2`はFCOS label `1,2,3`に連続化されます。公式1 Mpx評価に合わせ、既定で
先頭0.5秒と対角60 px未満のboxを除外します。配布版のclass定義が異なる場合は`--class-ids`
とconfigの`model.num_classes`（背景を含む）を合わせてください。

### DSEC / M3ED / MVSEC

各datasetもnative HDF5を直接index化します。`--search-root`は、そのsplitに含めたいsequenceだけが
置かれたdirectoryを指定します。

```bash
sla-index-h5 --dataset dsec --data-root /datasets/dsec \
  --search-root /datasets/dsec/train --output-dir /datasets/dsec/manifests --split train \
  --long-window-us 50000

sla-index-h5 --dataset m3ed --data-root /datasets/m3ed \
  --search-root /datasets/m3ed/train --output-dir /datasets/m3ed/manifests --split train \
  --long-window-us 50000

sla-index-h5 --dataset mvsec --data-root /datasets/mvsec \
  --search-root /datasets/mvsec/train --output-dir /datasets/mvsec/manifests --split train \
  --long-window-us 50000
```

```bash
sla-pretrain --config-name dsec
sla-pretrain --config-name m3ed
sla-pretrain --config-name mvsec
```

pretrainingの共通defaultはstudent側`data.short_window_us=10000`（10 ms）、teacher側
`data.long_window_us=50000`（50 ms）です。manifest生成時も`--long-window-us 50000`を指定し、
各sequence先頭でlong windowが欠けるsampleを除外します。

対応するnative pathはDSECの`/events/*`、M3EDの`/prophesee/left/*`、MVSECの
`/davis/left/events`です。right cameraはindex時の`--camera right`とconfigの
`data.event_camera: right`で選べます。

MVSECの学習前確認として、5 ms event windowをPNGへ描画できます。negative eventは赤、
positive eventは青で表示します。`--frames`を2以上にして`.gif`または`.mp4`を指定すると
時系列表示になります。

```bash
sla-visualize-events \
  --input /datasets/mvsec/outdoor_day/outdoor_day1_data.hdf5 \
  --output outputs/visualizations/mvsec_day1_5ms.png \
  --camera left --window-us 5000

sla-visualize-events \
  --input /datasets/mvsec/outdoor_day/outdoor_day1_data.hdf5 \
  --output outputs/visualizations/mvsec_day1.mp4 \
  --camera left --window-us 5000 --step-us 5000 --frames 100 --fps 20
```

`/data`だけを持つprecomputed tensor H5はraw event streamではありません。任意のshort/long
windowを再構成できないため、この実装の入力対象外です。1 Mpx H5にはtimestamp付きの
`x,y,t,p` event arraysが必要です。

NumPy 形式の小規模データは `split/class_name/*.npz` または `*.npy` を次で変換できます。
配列は `x,y,t,p`、structured array、または `[N,4]` の `events` を受け付けます。

```bash
sla-convert-numpy \
  --input-root /raw/sanity_check_events \
  --output-root /datasets/sanity_check_events \
  --height 128 --width 128 --timestamp-scale 1
```

詳細なschemaは[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)、dataset別の対応範囲と注意点は
[docs/DATASETS.md](docs/DATASETS.md)を参照してください。

## 実験手順

まず training split の密度分位点を確認し、config の `low_density_max` と
`high_density_min` に 1/3、2/3 quantile を設定します。

```bash
sla-data-stats --config configs/eval/data_stats.yaml
```

1 Mpx の SLA-SSL pretraining は次で実行します。

```bash
torchrun --standalone --nproc_per_node=4 -m slassl.cli.pretrain \
  --config-name prophesee_1mp \
  training.batch_size=2
```

続いて student encoder を 1 ms FCOS に移植し、評価します。

```bash
sla-finetune --config-name prophesee_1mp_detection \
  training.pretrained_checkpoint=/path/to/pretrain/checkpoint_last.pt
sla-evaluate --config configs/eval/prophesee_1mp_detection.yaml \
  --set checkpoint=/path/to/finetune/checkpoint_last.pt
```

M3ED/MVSECのdense downstream manifestはnative GT timestampから作ります。

```bash
sla-index-dense --dataset m3ed --task flow \
  --data-root "$M3ED_ROOT" --search-root "$M3ED_ROOT" \
  --output-dir outputs/m3ed/manifests_dense --split train \
  --include car_urban_day_penno_big_loop --include car_urban_day_penno_small_loop

sla-index-dense --dataset m3ed --task flow \
  --data-root "$M3ED_ROOT" --search-root "$M3ED_ROOT" \
  --output-dir outputs/m3ed/manifests_dense --split val \
  --include car_urban_day_rittenhouse

sla-index-dense --dataset m3ed --task segmentation \
  --data-root "$M3ED_ROOT" --search-root "$M3ED_ROOT" \
  --output-dir outputs/m3ed/manifests_dense --split train \
  --include car_urban_day_city_hall --include car_urban_day_penno_big_loop

sla-index-dense --dataset m3ed --task segmentation \
  --data-root "$M3ED_ROOT" --search-root "$M3ED_ROOT" \
  --output-dir outputs/m3ed/manifests_dense --split val \
  --include car_urban_day_ucity_small_loop

sla-index-dense --dataset mvsec --task flow \
  --data-root "$MVSEC_ROOT" --search-root "$MVSEC_ROOT/outdoor_day" \
  --output-dir outputs/mvsec/manifests_dense --split train \
  --include outdoor_day2 --end-fraction 0.8 --boundary-margin-us 50000

sla-index-dense --dataset mvsec --task flow \
  --data-root "$MVSEC_ROOT" --search-root "$MVSEC_ROOT/outdoor_day" \
  --output-dir outputs/mvsec/manifests_dense --split val \
  --include outdoor_day2 --start-fraction 0.8 --boundary-margin-us 50000

sla-index-dense --dataset mvsec --task flow \
  --data-root "$MVSEC_ROOT" --search-root "$MVSEC_ROOT/outdoor_day" \
  --output-dir outputs/mvsec/manifests_dense --split test \
  --include outdoor_day1
```

SSL student encoderをflow/segmentation decoderへ移植してfine-tuningします。

```bash
sla-finetune --config-name m3ed_flow \
  training.pretrained_checkpoint=/path/to/pretrain/checkpoint_last.pt \
  data.root="$M3ED_ROOT" \
  data.train_manifest="$PWD/outputs/m3ed/manifests_dense/train_flow.jsonl" \
  data.validation_manifest="$PWD/outputs/m3ed/manifests_dense/val_flow.jsonl"

sla-finetune --config-name m3ed_segmentation \
  training.pretrained_checkpoint=/path/to/pretrain/checkpoint_last.pt \
  data.root="$M3ED_ROOT" \
  data.train_manifest="$PWD/outputs/m3ed/manifests_dense/train_segmentation.jsonl" \
  data.validation_manifest="$PWD/outputs/m3ed/manifests_dense/val_segmentation.jsonl"

sla-finetune --config-name mvsec_flow \
  hydra.run.dir="$PWD/outputs/downstream/mvsec_flow_resnet50" \
  training.pretrained_checkpoint=/path/to/pretrain/checkpoint_last.pt \
  data.root="$MVSEC_ROOT" \
  data.train_manifest="$PWD/outputs/mvsec/manifests_dense/train_flow.jsonl" \
  data.validation_manifest="$PWD/outputs/mvsec/manifests_dense/val_flow.jsonl"
```

flow validationでは`all/aepe`を最小化するepochを`checkpoint_best.pt`として保存します。
`evaluation.every_epochs`ごとに固定6 sampleのevent、GT、予測、event-masked GT/予測、評価maskを
`validation_visualizations/epoch_XXXX/`とTensorBoardの`validation_flow/*`へ記録します。
MVSECの既定は毎epoch、M3EDの既定は計算量を考慮して5 epochごとです。

fine-tuned optical-flow checkpointは、F3と同じevent-supported protocolで可視化できます。
GTと予測にはsample内で共通のmagnitude scaleを使い、`event_masked`版にはfinite/nonzero GT、
dataset固有の有効領域、event supportの積を適用します。

```bash
sla-visualize-flow \
  --config configs/eval/mvsec_flow.yaml \
  --output-dir "$PWD/outputs/downstream/mvsec_flow_resnet50/visualizations/test" \
  --max-samples 100 \
  --sample-stride 10 \
  --set checkpoint="$PWD/outputs/downstream/mvsec_flow_resnet50/checkpoint_best.pt" \
  --set data.root="$MVSEC_ROOT" \
  --set data.manifest="$PWD/outputs/mvsec/manifests_dense/test_flow.jsonl" \
  --set data.flow_target_duration_us=null
```

出力は`prediction/{full,event_masked}`、`ground_truth/{full,event_masked}`、`events`、
`masks/{event_support,evaluation_valid}`に分かれます。`--save-arrays`を付けると可視化前の
flowとmaskも圧縮NPZで保存します。sequence間で色のmagnitude scaleを固定する場合は
`--max-flow 20`のように明示します。

label schema、11-class mapping、flow valid-mask protocolはApache-2.0のFast Feature Fields実装を
参照しています。F3本体、SegFormer、Transformers依存は取り込んでいません。

scratch 1 ms と scratch 5 ms は同じ fine-tuning 条件から作れます。

```bash
sla-finetune --config-name prophesee_1mp_detection \
  training.pretrained_checkpoint=null run_name=scratch_1ms

sla-finetune --config-name prophesee_1mp_detection \
  training.pretrained_checkpoint=null data.short_window_us=5000 run_name=scratch_5ms
```

`data.short_window_us=500` のような Hydra override で 0.5 / 1 / 2 / 5 / 10 ms curve を
作れます。一括実行は `bash scripts/run_accumulation_curve.sh`、loss component は
`bash scripts/run_ablations.sh`、Slurm template は `scripts/slurm/` にあります。実験条件一覧は
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) にまとめています。

## 出力

各runは既定で`outputs/train/YYYY-MM-DD/HH-MM-SS_run_name/`へ保存されます。`run_name`は
Hydra overrideで変更できます。各runには以下が含まれます。

- `resolved_config.json`: override 適用後の設定
- `metrics.jsonl`: step ごとの学習指標
- `validation_metrics.jsonl`: density subset を含む validation 指標
- `checkpoint_XXXX.pt`, `checkpoint_last.pt`: optimizer / scaler / scheduler を含む checkpoint
- `checkpoint_best.pt`, `best_validation.json`: validation monitorが最良のcheckpointと選択根拠
- `validation_visualizations/`: flow validationの固定sample可視化
- `tensorboard/`: train loss、各S2L scale、LR、EMA momentum、表現崩壊診断、validation metric

評価 JSON には accumulation time と GPU forward の sample 当たり latency も記録されます。

TensorBoardは複数runをまとめて起動できます。

```bash
tensorboard --logdir outputs/train
```

SSL pretrainingでは`train/diagnostics/*`にstudent/teacherのfeature・projection標準偏差、
L2正規化後の標準偏差、norm、student-teacher cosine、sample間cosineを記録します。標準偏差と
sample間cosineはbatch内のsample方向で計算するため、崩壊判定にはbatch size 2以上を使用し、
単一stepではなく推移を確認してください。`train/occupancy/*`には予測/target event率と
positive/negative probability、`train/optimization/*`にはgradient normとAMP scale、
`train/data/event_density`には入力event密度を記録します。

高解像度入力ではgradient accumulationでGPUへ載せるbatchを小さくできます。
`training.batch_size`は1回のforwardに使うmicro batch、
`training.gradient_accumulation_steps`は1回のoptimizer更新までに蓄積する回数です。

```bash
sla-pretrain --config-name m3ed \
  training.batch_size=2 \
  training.gradient_accumulation_steps=8
```

この例のeffective batch sizeは1 GPUで16、DistributedDataParallelでは`16 * world_size`です。
LR scheduler、EMA teacher、global step、`training.log_every`はoptimizer更新単位で進みます。
勾配は同等でもBatchNormの統計はmicro batch単位で計算されるため、物理batch 16との数値的な
完全一致は保証されません。TensorBoardの`train/optimization/effective_batch_size`と
`train/optimization/micro_batches_per_update`で実際の更新単位を確認できます。

## Sequence、recurrent、ViT

`data.sequence_length`を2以上にすると、同じrecording内の時間順windowを
`[batch,time,polarity,bins,height,width]`として読み込みます。GRU/LSTMは単方向で、未来の
windowを参照しません。ViT + GRU + multi-scale distillationの小規模設定は次で実行できます。

```bash
sla-pretrain --config-name sanity_check_experiment_vit_gru
sla-finetune --config-name sanity_check_classification_vit_gru \
  training.pretrained_checkpoint=/path/to/pretrain/checkpoint_last.pt
```

## 実装上の判断

1 Mpx で pixel-level occupancy をそのまま decode するとメモリ消費が大きいため、occupancy
target は `data.occupancy_stride` ごとの空間セルへ集約します。時間 bin と polarity channel は
保持します。これは実験変数なので stride を必ず記録してください。

Phase 1の分類、Phase 2の物体検出、およびDSEC/M3ED/MVSECを使ったPhase 3のSSL pretrainingは
実行可能です。Phase 3のsupervised optical flow、depth、segmentation headとlabel loaderはまだ
含めていません。native event readerとpretrained encoderはこれらdownstream taskで共有できます。
