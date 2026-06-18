# Data format

## Native sequence HDF5

HDF5を共通形式へコピーせず、各datasetのnative layoutを直接読みます。timestampはreader内で
microsecondsへ正規化され、昇順である必要があります。標準split layoutは次の形です。

```text
sequence.h5
└── events
    ├── x  uint16 [N]
    ├── y  uint16 [N]
    ├── t  int64  [N]  # microseconds
    └── p  int8   [N]  # 0/1 または -1/+1
```

| Dataset | Native event path | Timestamp handling |
|---|---|---|
| 1 Mpx H5 | auto-detected split/compound `x,y,t,p` | integer us by default |
| DSEC | `/events/{x,y,t,p}` | us + `/t_offset` |
| M3ED | `/prophesee/{left,right}/{x,y,t,p}` | `ms_map_idx`から単位を検証 |
| MVSEC | `/davis/{left,right}/events` | native matrixはseconds、F3 split版はusを自動判定 |

DSECの`/ms_to_idx`とM3EDの`/ms_map_idx`はtimestamp scaleの検証に使います。layoutやscaleが
異なる派生データではconfigの`event_group`、`timestamp_scale_to_us`、
`timestamp_offset_us`を明示してください。

F3形式の`50khz_*.npy`がsequence directoryにあれば、`sla-index-h5`が自動検出して20 usごとの
event index hintとしてmanifestへ登録します。これは検索高速化用で、存在しなくてもHDF5上の
binary searchへfallbackします。

`/data`のみのprecomputed tensor H5はraw event H5と異なるため対象外です。

`EventWindowDataset`はnative HDF5 dataset上でbinary searchし、sampleごとに必要なtimestamp
範囲だけを読みます。sequence 全体を RAM へロードしません。

## Manifest JSONL

SSL sample:

```json
{"sequence":"sequences/train/seq01.h5","timestamp_us":123456,"sample_id":"seq01:123456"}
```

Classification sample:

```json
{"sequence":"sequences/train/seq01.h5","timestamp_us":123456,"sample_id":"seq01:123456","label":3}
```

Detection sample (`boxes` は absolute XYXY、`labels` は背景 0 を避けて 1 始まり):

```json
{"sequence":"sequences/train/seq01.h5","timestamp_us":123456,"sample_id":"seq01:123456","boxes":[[10,20,40,80]],"labels":[1]}
```

1 MpxではbboxをJSONへ複製せず、structured NPYの該当sliceを参照します。

```json
{"sequence":"train/seq01.h5","timestamp_us":123456,"sample_id":"seq01:123456","annotations":"train/seq01_bbox.npy","annotation_start":20,"annotation_end":23}
```

`sequence` は config の `data.root` からの相対 path です。short と long の終端はどちらも
`timestamp_us` であり、未来のイベントは読みません。

## Voxel and density

voxel shape は `[polarity, temporal_bin, height, width]` です。model 入力時に前 2 軸を
flatten します。event density の単位は `events / pixel / ms` です。

occupancy target は `[polarity, temporal_bin, ceil(H/stride), ceil(W/stride)]` です。同じセルに
1 event 以上あれば 1、それ以外は 0 です。
