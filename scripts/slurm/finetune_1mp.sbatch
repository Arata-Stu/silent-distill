#!/usr/bin/env bash
#SBATCH --job-name=sla-finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/slurm
source .venv/bin/activate
sla-finetune --config configs/finetune/prophesee_1mp_detection.yaml

