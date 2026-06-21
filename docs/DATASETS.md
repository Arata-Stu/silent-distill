# Dataset support

## Support matrix

| Dataset | SSL event input | Current downstream | Native layout |
|---|---|---|---|
| Prophesee 1 Mpx | supported | FCOS detection | event H5 + bbox NPY |
| DSEC | supported | optical flow, test submission | `/events/{x,y,t,p}` |
| M3ED | supported | optical flow, semantic segmentation | `/prophesee/{left,right}/{x,y,t,p}` |
| MVSEC | supported | optical flow | `/davis/{left,right}/events` |

「supported」は、native HDF5からcausal short/long voxelを直接生成してSLA-SSL pretrainingを
実行できることを表します。dense taskはGT timestampへcausal event windowを合わせ、共通の
multi-scale decoderでfine-tuningします。depthとdisparityはまだ含まれません。

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

flow test adapterは公式`test_forward_flow_timestamps.csv`の各
`from_timestamp_us,to_timestamp_us,file_index`をそのまま使用します。入力windowは約100 msの
`[from,to]`、出力は同じ区間のforward displacementです。提出対象はCSV記載sampleだけで、
filenameは`file_index`を6桁zero paddingします。test GTは非公開なので、ローカルreportに精度指標は
出ません。DSEC serverがrectified left event-camera上のGT-valid全pixelでEPE、1PE、2PE、3PE、AEを
計算します。MVSEC用のevent-support maskや下端cropはDSECへ適用しません。
flow用SSLはstudent 100 ms、EMA teacher 200 msとし、studentを公式約100 ms intervalへ合わせます。

train/validation adapterは`forward_timestamps.txt`と同数の`flow/forward/*.png`を対応付けます。
PNG第3 channelだけをvalid maskとして使うため、GT flowが0の有効pixelも評価対象です。splitは
sequence-held-outを推奨し、`sla-index-dsec-flow --exclude <val-sequence>`と
`--include <val-sequence>`を対にして固定します。test sequenceをfine-tuningやmodel selectionへ
使用してはいけません。validation JSONは全pixelをまとめた`all`に加え、`per_sequence`と
sequenceを同じ重みで平均する`sequence_average`を出力し、best checkpointは後者のAEEで選びます。

Official reference:
https://dsec.ifi.uzh.ch/data-format/
https://dsec.ifi.uzh.ch/optical-flow-submission-format/
https://github.com/uzh-rpg/DSEC

## M3ED

processed `data.h5`はtime-synchronizedかつdecoded済みです。eventは
`/prophesee/{left,right}`にあり、`ms_map_idx`をtimestamp単位の検証に使います。event座標は
distortedなので、dense taskではH5に埋め込まれたcalibrationを使用します。depth、pose、
semanticsは別H5にあります。

flowは`flow/prophesee/left/{x,y}`と`ts`、semantic pseudo-labelは`predictions`と`ts`を読みます。
semanticはCityscapes 19 classからDSEC互換11 classへ変換し、ignore labelは255です。

Official references:
https://m3ed.io/data_overview/datafiles/
https://github.com/daniilidis-group/m3ed

## MVSEC

official ROS-free HDF5の`/davis/{left,right}/events`は`[x,y,t,p]` matrixで、timestampはsecondsから
microsecondsへ変換します。F3の`process_mvsec.py`でsplitされた`events/{x,y,t,p}`版は既にusなので、
readerがlayoutとdtypeから自動判定します。flowはground-truth HDF5の
`/davis/left/flow_dist`と`flow_dist_ts`を読みます。`flow[i]`には`[ts[i],ts[i+1]]`のeventを
対応させ、finite・nonzero・event supportを持つpixelをprimary評価に使います。画像下端のcropは
outdoor sequenceだけに適用し、GT-valid dense評価もsecondaryとして保存します。

Official reference:
https://daniilidis-group.github.io/mvsec/download/

## Split policy

sequence由来の近接sampleがtrain/validation/testをまたがないよう、sequence単位でsplitを作ります。
`sla-index-h5`の`--search-root`には対象splitのsequenceだけを含めてください。event-density thresholdは
training splitだけから推定します。

MVSECのように独立sequence数が不足する場合は、test sequenceを触らず、training sequenceを
`sla-index-dense --end-fraction 0.8`と`--start-fraction 0.8`で時間順にtrain/validationへ分けます。
両方に`--boundary-margin-us 50000`を指定すると境界の両側50 msを除外し、近接windowの重複を
抑制できます。このvalidationはsequence-held-outではなくtemporal holdoutとして報告します。
