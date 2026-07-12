#!/bin/bash
#SBATCH --job-name=video_swin
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/src/sailsprep/action_model_testing/video_swin/clip_based/logs/video_swin_%A_%a_%x.out
#SBATCH --error=/src/sailsprep/action_model_testing/video_swin/clip_based/logs/video_swin_%A_%a_%x.err
#SBATCH --array=0-2

# ==========================================================================
# Multi-seed fine-tuning for Video Swin-B on a single task.
#
# Array index mapping (3 seeds):
#   0 -> seed 42
#   1 -> seed 123
#   2 -> seed 456
#
# Usage:
#   sbatch job_clip.sh loco      # 3 jobs: loco x seeds 42, 123, 456
#   sbatch job_clip.sh rmm       # 3 jobs: rmm  x seeds 42, 123, 456
#
# ==========================================================================

LABEL=${1:?"ERROR: pass task as first arg, e.g.: sbatch job_clip.sh loco"}
if [[ "${LABEL}" != "loco" && "${LABEL}" != "rmm" ]]; then
    echo "ERROR: argument must be 'loco' or 'rmm', got: ${LABEL}"
    exit 1
fi

# Derive seed from array index.
SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR=/src/sailsprep/action_model_testing/video_swin/clip_based
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# --- Modules + conda ---
module load miniforge
module load cuda
module load cudnn
conda deactivate 2>/dev/null || true
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate pytorchvideo_env

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Array task   : ${SLURM_ARRAY_TASK_ID}"
echo "Node         : ${SLURMD_NODENAME}"
echo "Task         : ${LABEL}"
echo "Seed         : ${SEED}"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory   : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Start time   : $(date)"
echo "=========================================="

cd "${SCRIPT_DIR}"

python video_swin_finetune.py --task "${LABEL}" --seed "${SEED}"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="