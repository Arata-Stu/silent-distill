#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pretrain/prophesee_1mp.yaml}"

sla-pretrain --config "$CONFIG" --set output_dir=outputs/ablation/s2l_only \
  --set loss.lambda_silence=0.0
sla-pretrain --config "$CONFIG" --set output_dir=outputs/ablation/silence_only \
  --set loss.lambda_s2l=0.0 --set loss.lambda_silence=1.0
sla-pretrain --config "$CONFIG" --set output_dir=outputs/ablation/s2l_silence \
  --set loss.lambda_silence=1.0
sla-pretrain --config "$CONFIG" --set output_dir=outputs/ablation/with_polarity \
  --set loss.lambda_polarity=0.1
sla-pretrain --config "$CONFIG" --set output_dir=outputs/ablation/with_rate \
  --set loss.lambda_polarity=0.1 --set loss.lambda_rate=0.1
