# Experiment matrix

各 run で seed、GPU 数、global batch size、data split、density threshold、occupancy stride を
固定・記録してください。比較する fine-tuning run は optimizer schedule も同一にします。

## Main comparison

| Run | Pretraining | Inference window | Command change |
|---|---|---:|---|
| Scratch-short | none | 1 ms | `training.pretrained_checkpoint=null` |
| Scratch-long | none | 5 ms | 上記 + `data.short_window_us=5000` |
| S2L | S2L only | 1 ms | `loss.lambda_silence=0` |
| SLA-SSL | S2L + silence | 1 ms | default |

外部の MEM / MAE、F3 checkpoint と比較するときは、同じ ResNet backbone、voxel、split、
fine-tuning config に揃えた encoder state を使用してください。外部手法そのものの再実装は
このリポジトリには含めません。

## Accumulation robustness

fine-tuning と評価の両方で `data.short_window_us` を `500, 1000, 2000, 5000, 10000` に変更し、
mAP と入力窓長を保存します。SLA-SSL の主条件は 1 ms student / 5 ms teacher です。

## Loss components

| Condition | `lambda_s2l` | `lambda_silence` | `lambda_polarity` | `lambda_rate` |
|---|---:|---:|---:|---:|
| A | 1 | 0 | 0 | 0 |
| B | 0 | 1 | 0 | 0 |
| C | 1 | 1 | 0 | 0 |
| D | 1 | 1 | 0.1 | 0 |
| E | 1 | 1 | 0.1 | 0.1 |

## Other ablations

```bash
# student / teacher window
--set data.short_window_us=500 --set data.long_window_us=5000
--set data.short_window_us=1000 --set data.long_window_us=10000

# temporal bins and shuffled order
--set data.bins=1
--set data.bins=10
--set data.temporal_shuffle=true

# polarity
--set data.use_polarity=false --set loss.lambda_polarity=0

# negative sampling
--set 'loss.negative_modes=[random]'
--set 'loss.negative_modes=[near]'
--set 'loss.negative_modes=[random,near,hard]'
```

shell が bracket を展開しないよう、list override は quote してください。polarity や bins を
変えた pretraining checkpoint は同じ入力 channel 数の downstream config と組み合わせます。

## Density subsets

`sla-data-stats` の training split 1/3・2/3 quantile を low/medium/high の境界に使います。
test split から閾値を推定しません。main metric に加え subset ごとの sample 数も報告します。

## Cross-dataset pretraining

`configs/pretrain/{dsec,m3ed,mvsec}.yaml`は各native HDF5 layoutを直接読みます。sensor resolutionが
異なるため、現状はdatasetごとにrunを分け、同じbackboneのcheckpointを下流taskへ移植します。
optical flow、depth、segmentationのsupervised headは未実装なので、Phase 3の現対応範囲は
representation pretrainingとevent-density評価用manifestの生成までです。
