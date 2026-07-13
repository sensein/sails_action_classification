#!/bin/bash
#SBATCH --job-name=IV2_finetune
#SBATCH --partition=ou_bcs_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/src/sailsprep/action_model_testing/internvideo2/logs/IV2_%A_%a_%x.out
#SBATCH --error=/src/sailsprep/action_model_testing/internvideo2/logs/IV2_%A_%a_%x.err
#SBATCH --array=0-2
# ==========================================================================
# Multi-seed fine-tuning for a single task.
#
# Array index -> seed mapping:
#   0 -> seed 42
#   1 -> seed 123
#   2 -> seed 456
#
# Usage:
#   sbatch internvideo2_finetune.sh loco      # 3 array jobs: loco x seeds 42, 123, 456
#   sbatch internvideo2_finetune.sh rmm       # 3 array jobs: rmm  x seeds 42, 123, 456
#

LABEL=${1:?"ERROR: pass task as first arg, e.g.: sbatch internvideo2_finetune.sh loco"}
if [[ "${LABEL}" != "loco" && "${LABEL}" != "rmm" ]]; then
    echo "ERROR: argument must be 'loco' or 'rmm', got: ${LABEL}"
    exit 1
fi


SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR=/src/sailsprep/action_model_testing/internvideo2
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

module load miniforge
module load cuda
module load cudnn
conda deactivate 2>/dev/null || true
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate iv2

export HF_HOME=/orcd/data/satra/002/huggingface
export HUGGINGFACE_HUB_CACHE=/orcd/data/satra/002/huggingface/hub
export TRANSFORMERS_CACHE=/orcd/data/satra/002/huggingface/hub


export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

NUM_GPUS=1

echo "=========================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Array task   : ${SLURM_ARRAY_TASK_ID}"
echo "Node         : ${SLURMD_NODENAME}"
echo "Task         : ${LABEL}"
echo "Seed         : ${SEED}"
echo "Num GPUs     : ${NUM_GPUS}"
echo "GPU info     :"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Python       : $(which python)"
echo "HF_HOME      : ${HF_HOME}"
echo "Start time   : $(date)"
echo "=========================================="

cd "${SCRIPT_DIR}"

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    internvideo2_finetune.py \
    --task "${LABEL}" \
    --seed "${SEED}" \
    --gpus "${NUM_GPUS}"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="