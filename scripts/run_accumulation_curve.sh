#!/usr/bin/env bash
set -euo pipefail

FINETUNE_CONFIG_NAME="${1:-prophesee_1mp_detection}"
EVAL_CONFIG="${2:-configs/eval/prophesee_1mp_detection.yaml}"
RUN_PREFIX="${3:-sla}"
PRETRAINED_CHECKPOINT="${4:-null}"

for WINDOW_US in 500 1000 2000 5000 10000; do
  RUN_DIR="outputs/accumulation/${RUN_PREFIX}_${WINDOW_US}us"
  sla-finetune --config-name "$FINETUNE_CONFIG_NAME" \
    hydra.run.dir="$RUN_DIR" run_name="${RUN_PREFIX}_${WINDOW_US}us" \
    data.short_window_us="$WINDOW_US" \
    training.pretrained_checkpoint="$PRETRAINED_CHECKPOINT"
  sla-evaluate --config "$EVAL_CONFIG" \
    --set data.short_window_us="$WINDOW_US" \
    --set checkpoint="$RUN_DIR/checkpoint_last.pt" \
    --set output_file="$RUN_DIR/test_metrics.json"
done
