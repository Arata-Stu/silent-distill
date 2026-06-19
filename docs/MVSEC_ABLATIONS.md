# MVSEC SSL ablation protocol

## Main question

最初のablationでは「事前学習が効くか」と「SLA-SSLの主要objectiveのどちらが効くか」だけを
切り分けます。polarity、rate、negative sampling、multi-scaleは主表の結果を確認してから行います。

| ID | Pretraining | S2L | Silence | Reconstruction |
|---|---|---:|---:|---:|
| Scratch | なし | 0 | 0 | 0 |
| AE | short-voxel AutoEncoder | 0 | 0 | 1 |
| S2L | SLA-SSL | 1 | 0 | 0 |
| Silence | SLA-SSL | 0 | 1 | 0 |
| SLA | SLA-SSL | 1 | 1 | 0 |

Scratchはlossを全て0にしてpretrainingする条件ではありません。random initializationから直接
Flow fine-tuningします。AEはshort voxelの`log1p(count)`を再構成し、SLA-SSLと同じencoderだけを
downstreamへ移します。

## Data policy

- SSL / AE pretraining: `outdoor_day2`
- SSL / AE pretraining範囲: `outdoor_day2`の先頭80%のみ
- Flow train: `outdoor_day2`の先頭80%
- Flow validation: `outdoor_day2`の末尾20%
- Flow test: `outdoor_day1`
- temporal split境界の両側50 msは除外
- `outdoor_day1`はpretraining、hyperparameter選択、checkpoint選択に使用しない

この主表のFlow入力は10 msで、SLA-SSLのshort-window性能を見るlatency-stress ablationです。
MVSEC native flow displacementの時間幅全体を観測するSOTA protocolとは分けて扱います。主表で上位の
Scratch / AE / S2L / SLA条件は、50 ms入力でも再評価し、「SSL objectiveの効果」と「短時間入力の
難しさ」を切り分けます。F3の45 Hz / 11.25 Hz表との直接比較には、さらにF3固有のwindow、crop、
flow scaling、集計方法を揃える必要があります。

以前の`outdoor_day1` SSL checkpointを`outdoor_day1` Flow testへ使う場合はtransductive pretraining
として別表に分けます。SOTAとの通常比較には使いません。

## Manifest preparation

```bash
cd "$HOME/project/research/silent-distill"
python -m pip install -e .

export MVSEC_ROOT=/media/arata-24/AT_SSD/dataset/mvsec
export ABLATION_ROOT="$PWD/outputs/ablations/mvsec"
export SSL_MANIFEST_DIR="$ABLATION_ROOT/manifests_ssl"
export FLOW_MANIFEST_DIR="$ABLATION_ROOT/manifests_flow"

mkdir -p "$SSL_MANIFEST_DIR" "$FLOW_MANIFEST_DIR"

sla-index-h5 \
  --dataset mvsec \
  --data-root "$MVSEC_ROOT" \
  --search-root "$MVSEC_ROOT/outdoor_day" \
  --file-glob 'outdoor_day2_data.hdf5' \
  --output-dir "$SSL_MANIFEST_DIR" \
  --split pretrain \
  --camera left \
  --sample-period-us 5000 \
  --long-window-us 50000 \
  --end-fraction 0.8 \
  --boundary-margin-us 50000

sla-index-dense \
  --dataset mvsec --task flow \
  --data-root "$MVSEC_ROOT" \
  --search-root "$MVSEC_ROOT/outdoor_day" \
  --output-dir "$FLOW_MANIFEST_DIR" \
  --split train --include outdoor_day2 \
  --short-window-us 10000 \
  --end-fraction 0.8 --boundary-margin-us 50000

sla-index-dense \
  --dataset mvsec --task flow \
  --data-root "$MVSEC_ROOT" \
  --search-root "$MVSEC_ROOT/outdoor_day" \
  --output-dir "$FLOW_MANIFEST_DIR" \
  --split val --include outdoor_day2 \
  --short-window-us 10000 \
  --start-fraction 0.8 --boundary-margin-us 50000

sla-index-dense \
  --dataset mvsec --task flow \
  --data-root "$MVSEC_ROOT" \
  --search-root "$MVSEC_ROOT/outdoor_day" \
  --output-dir "$FLOW_MANIFEST_DIR" \
  --split test --include outdoor_day1 \
  --short-window-us 10000
```

## Pretraining commands

まずseed 42でpilotを行います。全条件でepoch、optimizer、batch、augmentationを変えません。
これはencoderのoptimizer update数を揃える比較であり、teacherを持つSLAとdecoderを持つAEのFLOPsは
同一ではありません。GPU時間とpeak memoryも併記します。

```bash
sla-pretrain --config-name mvsec \
  hydra.run.dir="$ABLATION_ROOT/seed42/s2l/pretrain" \
  run_name=mvsec_ablation_s2l_seed42 \
  seed=42 \
  data.root="$MVSEC_ROOT" \
  data.manifest="$SSL_MANIFEST_DIR/pretrain_ssl.jsonl" \
  loss.lambda_s2l=1.0 loss.lambda_silence=0.0 \
  loss.lambda_polarity=0.0 loss.lambda_rate=0.0

sla-pretrain --config-name mvsec \
  hydra.run.dir="$ABLATION_ROOT/seed42/silence/pretrain" \
  run_name=mvsec_ablation_silence_seed42 \
  seed=42 \
  data.root="$MVSEC_ROOT" \
  data.manifest="$SSL_MANIFEST_DIR/pretrain_ssl.jsonl" \
  loss.lambda_s2l=0.0 loss.lambda_silence=1.0 \
  loss.lambda_polarity=0.0 loss.lambda_rate=0.0

sla-pretrain --config-name mvsec \
  hydra.run.dir="$ABLATION_ROOT/seed42/sla/pretrain" \
  run_name=mvsec_ablation_sla_seed42 \
  seed=42 \
  data.root="$MVSEC_ROOT" \
  data.manifest="$SSL_MANIFEST_DIR/pretrain_ssl.jsonl" \
  loss.lambda_s2l=1.0 loss.lambda_silence=1.0 \
  loss.lambda_polarity=0.0 loss.lambda_rate=0.0

sla-pretrain --config-name mvsec_autoencoder \
  hydra.run.dir="$ABLATION_ROOT/seed42/autoencoder/pretrain" \
  run_name=mvsec_ablation_autoencoder_seed42 \
  seed=42 \
  data.root="$MVSEC_ROOT" \
  data.manifest="$SSL_MANIFEST_DIR/pretrain_ssl.jsonl"
```

## Flow fine-tuning

各pretraining checkpointに対して、同じcommandの`CONDITION`だけを変えます。

```bash
export CONDITION=sla
export SSL_CKPT="$ABLATION_ROOT/seed42/$CONDITION/pretrain/checkpoint_last.pt"
export FLOW_RUN="$ABLATION_ROOT/seed42/$CONDITION/flow"

sla-finetune --config-name mvsec_flow \
  hydra.run.dir="$FLOW_RUN" \
  run_name="mvsec_flow_${CONDITION}_seed42" \
  seed=42 \
  training.pretrained_checkpoint="$SSL_CKPT" \
  data.root="$MVSEC_ROOT" \
  data.train_manifest="$FLOW_MANIFEST_DIR/train_flow.jsonl" \
  data.validation_manifest="$FLOW_MANIFEST_DIR/val_flow.jsonl"
```

`CONDITION`は`s2l`、`silence`、`sla`、`autoencoder`の4条件で繰り返します。Scratchだけは次です。

```bash
export FLOW_RUN="$ABLATION_ROOT/seed42/scratch/flow"

sla-finetune --config-name mvsec_flow \
  hydra.run.dir="$FLOW_RUN" \
  run_name=mvsec_flow_scratch_seed42 \
  seed=42 \
  training.pretrained_checkpoint=null \
  data.root="$MVSEC_ROOT" \
  data.train_manifest="$FLOW_MANIFEST_DIR/train_flow.jsonl" \
  data.validation_manifest="$FLOW_MANIFEST_DIR/val_flow.jsonl"
```

## Final test evaluation

validation AEEで選ばれた`checkpoint_best.pt`だけを`outdoor_day1`で一度評価します。

```bash
export CONDITION=sla
export FLOW_RUN="$ABLATION_ROOT/seed42/$CONDITION/flow"

sla-evaluate \
  --config configs/eval/mvsec_flow.yaml \
  --set checkpoint="$FLOW_RUN/checkpoint_best.pt" \
  --set output_file="$FLOW_RUN/test_metrics.json" \
  --set data.root="$MVSEC_ROOT" \
  --set data.manifest="$FLOW_MANIFEST_DIR/test_flow.jsonl" \
  --set data.flow_target_duration_us=null
```

主表にはAEPE、1PE、2PE、3PE、AAE、model latencyを載せます。pilotで実行系を確定した後、
seed 42/43/44で同じ実験を行い、平均と標準偏差を報告します。

## Second-stage ablations

主表の後に以下を一項目ずつ変更します。

| Condition | Override |
|---|---|
| SLA + polarity | `loss.lambda_polarity=0.1` |
| SLA + rate | `loss.lambda_rate=0.1` |
| global S2L only | `model.multi_scale_distillation=false` |
| random negatives | `'loss.negative_modes=[random]'` |
| near negatives | `'loss.negative_modes=[near]'` |
| hard negatives | `'loss.negative_modes=[random,near,hard]'` |

全組合せを一度に回すのではなく、Scratch / AE / S2L / Silence / SLAの主表を確定してから進めます。
