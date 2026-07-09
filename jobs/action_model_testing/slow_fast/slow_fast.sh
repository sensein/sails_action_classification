#!/bin/bash
#SBATCH --job-name=slowfast_${LABEL}
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/src/sailsprep/action_model_testing/slow_fast/logs/slowfast_%j_%x.out
#SBATCH --error=/src/sailsprep/action_model_testing/slow_fast/logs/slowfast_%j_%x.err

# ============================================================
# Usage:
#   sbatch job.sh loco
#   sbatch job.sh rmm
# ============================================================

LABEL=${1:-loco}   # default: loco if no argument given

# Validate argument
if [[ "$LABEL" != "loco" && "$LABEL" != "rmm" ]]; then
    echo "ERROR: argument must be 'loco' or 'rmm', got: $LABEL"
    exit 1
fi

SCRIPT_DIR=/src/sailsprep/action_model_testing/slow_fast
LOG_DIR=${SCRIPT_DIR}/logs

mkdir -p ${LOG_DIR}

module load miniforge
module load cuda
module load cudnn
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate pytorchvideo_env

echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Label mode: $LABEL"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Start time: $(date)"
echo "=========================================="

cd ${SCRIPT_DIR}

python slow_fast.py --label ${LABEL}

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="