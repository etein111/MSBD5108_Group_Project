#!/bin/bash
#SBATCH --job-name=jevbench-infer
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/5018/outputs/infer_%j.out

set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate jevbench
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/mscbdtspod2026/cyeah/5018/hf

PROJ=/scratch/mscbdtspod2026/cyeah/5018
cd "$PROJ/code"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m block3.cluster_run \
  --input "$PROJ/code/evaluation/data/eval.jsonl" \
  --out-dir "$PROJ/outputs/preds-shared-shots" \
  --cache-dir "$HF_HOME" \
  --shots-file "$PROJ/code/evaluation/data/train.jsonl" \
  "$@"
