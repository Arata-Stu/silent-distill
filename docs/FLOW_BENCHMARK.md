# Optical-flow benchmark protocol

## DSEC official contract

DSEC testの比較で固定される条件は次の通りです。

- rectified left event-camera座標のforward optical flow
- CSVに記載されたreference timestampからtarget timestampまでのdisplacement（約100 ms）
- CSV指定sampleは約2 Hzで、全指定sampleを提出
- GT-validな全pixelでEPE、1PE、2PE、3PE、AEを計算
- test全体で同じparameterを使用し、test-time training / fine-tuningは禁止
- 16-bit RGB PNGは`flow * 128 + 2^15`、第3 channelは0または1
- serverが個別sequenceとall-sequence averageを計算

test GTは非公開なので、ローカルでtest metricを計算したという結果はDSEC公式評価ではありません。

## Reference implementation audit

| Repository | Relevant dataset | Useful reference | Important difference |
|---|---|---|---|
| E-RAFT | DSEC, MVSEC | DSEC test layout, `t_offset`, rectification, output indices, PNG writer | DSEC入力はreference前後のold/new 100 ms volume、15 bins。warm startも別条件 |
| EVA-Flow | DSEC | train/test timestamp pairs, flow decoder, rectification, single-volume variants | local cloneにcomplete submission exporterはない |
| E-STMFlow | DSEC | test layout、`t_offset`、rectification、100 ms interval、CSV-selected output、PNG writer | 100 msをsigned 32-bin voxel化。binをspatio-temporal sequenceとして処理するが、sample間のrecurrent stateはない。test intervalはimage timestampから再構成 |
| ADMFlow | MVSEC | `dt1`/`dt4` GT生成、valid range、sparse/dense masks | DSEC adapterではない。outdoor cropは190 rows、sample平均 |
| Fast Feature Fields | MVSEC, M3ED | F3 flow protocol、full/event-masked visualization | MVSECはcentered 50 ms、outdoor crop 193 rows、45/11.25 Hz scaling |

同じ`AEE`や`3PE`という名前でも、mask、crop、flow duration、sample/sequence/pixel aggregationが
異なれば数値は直接比較できません。

## Protocol used in this repository

### DSEC train / validation

- official `forward_timestamps.txt`と`flow/forward/*.png`をindex順に対応付ける
- eventを`rectify_map`でrectified left-camera座標へ移す
- 入力は各flow intervalの`[from,to)`だけを使うevent-only single window
- flow displacementはnative約100 msのまま使い、時間scaleしない
- valid maskはPNG第3 channelのみ。zero-flowのvalid pixelも含む
- sequence-held-out validationを使い、test sequenceはmodel selectionに使わない
- `all`は全valid pixel集計、`sequence_average`はsequence metricのmacro average

defaultは5 temporal binsです。これは既存SLA-SSL checkpointの入力channelと一致させるためで、
E-RAFTの15 binsやE-STMFlowのsigned・normalized 32 binsと同じ入力表現ではありません。
binsやpolarity表現、normalizationを変えた比較は別runとして記録します。

### DSEC test

- official CSVの全7 sequence・全rowをmanifest化
- `[from,to)` eventだけを使い、CSVのfile indexで`xxxxxx.png`を命名
- submission directoryを読み戻して16-bit RGB、shape、file集合を検証
- sequence directoryだけを含むZIPを作成
- test reportのmetricは`null`。精度はDSEC serverの結果だけを採用

## Fair comparison tracks

Published leaderboardとの比較では、最低限次を表へ併記します。

- event-onlyかglobal-shutter image併用か
- supervised DSEC flow GT、third-party GT、self-supervisedのどれを使ったか
- reference前のcontext、`[from,to]` interval、recurrent warm startの有無
- temporal bins、event representation、resolution
- parameter count、model-only latency、hardware、batch size
- server submission ID/dateとEPE、1PE、2PE、3PE、AE

SLA-SSL encoderの公平なablationでは、downstream architecture、DSEC train/validation split、
100 ms single-window入力、optimizer、epoch、checkpoint criterionを固定し、pretrainingだけを変更します。
E-RAFT warm-startや2-window入力との比較はofficial leaderboard比較としては有効ですが、encoder
pretrainingのcontrolled ablationとは分けて報告します。
