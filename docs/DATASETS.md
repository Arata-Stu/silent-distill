# Dataset support

## Support matrix

| Dataset | SSL event input | Current downstream | Native layout |
|---|---|---|---|
| Prophesee 1 Mpx | supported | FCOS detection | event H5 + bbox NPY |
| DSEC | supported | not yet implemented | `/events/{x,y,t,p}` |
| M3ED | supported | not yet implemented | `/prophesee/{left,right}/{x,y,t,p}` |
| MVSEC | supported | not yet implemented | `/davis/{left,right}/events` |

「supported」は、native HDF5からcausal short/long voxelを直接生成してSLA-SSL pretrainingを
実行できることを表します。DSEC/M3ED/MVSECのflow、depth、disparity、segmentation教師ラベルを
読むloaderとtask headはまだ含まれません。

## Prophesee 1 Mpx

想定入力はraw event arrays `x,y,t,p`を保持するH5と、official structured bbox NPYです。
H5内pathは自動判定されます。official toolboxに合わせて主要3クラスを評価し、先頭0.5秒と
小boxをfilterします。

H5が`/data`だけを持つprecomputed tensorの場合、0.5/1/2/5/10 msを任意に再voxelizeできません。
その形式は現状の対象外です。

Official reference:
https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox

## DSEC

event timestampはmicrosecondsで、`/t_offset`を加えて他sensorのclockに合わせます。H5のBlosc
compressionは`hdf5plugin`で読みます。SSLではdistorted event座標をそのまま使用できますが、
flow/disparity/segmentation評価ではofficial `rectify_maps.h5`による座標変換が必要です。

Official reference:
https://dsec.ifi.uzh.ch/data-format/

## M3ED

processed `data.h5`はtime-synchronizedかつdecoded済みです。eventは
`/prophesee/{left,right}`にあり、`ms_map_idx`をtimestamp単位の検証に使います。event座標は
distortedなので、dense taskではH5に埋め込まれたcalibrationを使用します。depth、pose、
semanticsは別H5にあります。

Official references:
https://m3ed.io/data_overview/datafiles/
https://github.com/daniilidis-group/m3ed

## MVSEC

official ROS-free HDF5の`/davis/{left,right}/events`は`[x,y,t,p]` matrixで、timestampはsecondsから
microsecondsへ変換します。F3の`process_mvsec.py`でsplitされた`events/{x,y,t,p}`版は既にusなので、
readerがlayoutとdtypeから自動判定します。depth/poseはground-truth HDF5、optical flowはofficial
toolで生成したNPZを使う想定ですが、現状はevent SSL inputだけに対応しています。

Official reference:
https://daniilidis-group.github.io/mvsec/download/

## Split policy

sequence由来の近接sampleがtrain/validation/testをまたがないよう、sequence単位でsplitを作ります。
`sla-index-h5`の`--search-root`には対象splitのsequenceだけを含めてください。event-density thresholdは
training splitだけから推定します。
