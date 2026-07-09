#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-14
#SBATCH --output=${CODE_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video/rmm}/logs/ablation_%A_%a.out
#SBATCH --error=${CODE_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video/rmm}/logs/ablation_%A_%a.err

CODE_DIR="${CODE_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video/rmm}"
export HF_HOME="${HF_HOME:-/home/aparnabg/.cache/huggingface}"
export PYTHONUNBUFFERED=1

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate vjepa2-312

# -------------------------------------------------------
# Map array index (0-14) -> (seed, head)
#   0  = seed 42,  linear        5  = seed 456, linear
#   1  = seed 42,  mlp_small     6  = seed 456, mlp_small
#   2  = seed 42,  mlp_large     7  = seed 456, mlp_large
#   3  = seed 42,  attentive     8  = seed 456, attentive
#   4  = seed 42,  transformer   9  = seed 456, transformer
#                               10  = seed 123, linear
#                               11  = seed 123, mlp_small
#                               12  = seed 123, mlp_large
#                               13  = seed 123, attentive
#                               14  = seed 123, transformer
# -------------------------------------------------------

SEEDS=(42 456 123)
HEADS=(linear mlp_small mlp_large attentive transformer)

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
HEAD_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

SEED=${SEEDS[$SEED_IDX]}
HEAD=${HEADS[$HEAD_IDX]}

echo "=============================================="
echo "Job array id : ${SLURM_ARRAY_TASK_ID}"
echo "Seed         : ${SEED}"
echo "Head         : ${HEAD}"
echo "Start        : $(date)"
echo "Node         : ${SLURMD_NODENAME}"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================="

python "${CODE_DIR}/train_probe_ablation.py" \
    --seed ${SEED} \
    --head ${HEAD}

echo "Done  seed=${SEED}  head=${HEAD}: $(date)"