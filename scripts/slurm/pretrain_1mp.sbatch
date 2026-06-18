#!/usr/bin/env bash
#SBATCH --job-name=sla-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/slurm
source .venv/bin/activate
torchrun --standalone --nproc_per_node=4 -m slassl.cli.pretrain \
  --config configs/pretrain/prophesee_1mp.yaml \
  --set training.batch_size=2

