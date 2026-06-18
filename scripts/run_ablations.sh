#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:-prophesee_1mp}"

sla-pretrain --config-name "$CONFIG_NAME" run_name=s2l_only loss.lambda_silence=0.0
sla-pretrain --config-name "$CONFIG_NAME" run_name=silence_only \
  loss.lambda_s2l=0.0 loss.lambda_silence=1.0
sla-pretrain --config-name "$CONFIG_NAME" run_name=s2l_silence loss.lambda_silence=1.0
sla-pretrain --config-name "$CONFIG_NAME" run_name=with_polarity loss.lambda_polarity=0.1
sla-pretrain --config-name "$CONFIG_NAME" run_name=with_rate \
  loss.lambda_polarity=0.1 loss.lambda_rate=0.1
