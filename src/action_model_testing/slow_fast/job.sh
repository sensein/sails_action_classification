#!/bin/bash
#SBATCH --job-name=slowfast_${LABEL}
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/h5_file/clip/logs/slowfast_%j_%x.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/h5_file/clip/logs/slowfast_%j_%x.err

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

SCRIPT_DIR=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/h5_file/clip
LOG_DIR=${SCRIPT_DIR}/logs

mkdir -p ${LOG_DIR}

module load miniforge
module load cuda
module load cudnn
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
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

python clips_combined.py --label ${LABEL}

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="