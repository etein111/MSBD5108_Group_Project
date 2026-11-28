#!/bin/bash
#SBATCH --job-name=jevbench-v2
#SBATCH --partition=normal
#SBATCH --account=mscbdtspod2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/mscbdtspod2026/cyeah/5018/outputs/v2_%j.out

# Usage:
#   sbatch slurm_v2.slurm <input.jsonl> <out-dir> [shots.jsonl]
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate jevbench
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/mscbdtspod2026/cyeah/5018/hf

PROJ=/scratch/mscbdtspod2026/cyeah/5018
INPUT=${1:?need input jsonl}
OUTDIR=${2:?need output dir}
SHOTS=${3:-$PROJ/code/evaluation/data_v2/train.jsonl}

cd "$PROJ/code"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "input=$INPUT out=$OUTDIR shots=$SHOTS"

python -m pipeline.cluster_run \
  --input "$INPUT" \
  --out-dir "$OUTDIR" \
  --cache-dir "$HF_HOME" \
  --shots-file "$SHOTS" \
  --shots-per-type 4
