#!/bin/bash
#SBATCH --job-name=entitysam_batch
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-4
#SBATCH --output=logs/entitysam_%A_%a.out
#SBATCH --error=logs/entitysam_%A_%a.err
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:2

# --- Setup ---
mkdir -p logs
set -euo pipefail

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "=========================================="

# --- Load and Initialize Conda ---
module load miniforge/23.11.0-0
CONDA_SH="${CONDA_SH:-/orcd/software/community/001/rocky8/miniforge/23.11.0-0/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
eval "$(conda shell.bash hook)"
conda activate batch_env

# Ensure environment site-packages are visible safely
export PYTHONPATH="$CONDA_PREFIX/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SAM2_DIR="${SAM2_DIR:-/home/aparnabg/sam2}"
export PYTHONPATH="${SAM2_DIR}/entitysam:${SAM2_DIR}:${PYTHONPATH:-}"


# --- Environment Diagnostics (Optional, safe checks) ---
echo "Python executable: $(which python)"
python --version
python -c "import sys; print('Sys.path:', sys.path)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import hydra; print('Hydra OK')"
python -c "import natsort; print('natsort OK')"
python -c "import tqdm; print('tqdm OK')"
python -c "import detectron2; print('detectron2 OK')"

nvidia-smi || echo "nvidia-smi not found"

# --- Configurations ---
CSV_FILE="${CSV_FILE:-/home/aparnabg/orcd/scratch/csvfiles/mutichild_in_out_paths.csv}"
CKPT_DIR="${CKPT_DIR:-${SAM2_DIR}/checkpoints/vit-s/}"
MODEL_CFG="configs/sam2.1_hiera_s.yaml"
MASK_DECODER_DEPTH=5
TARGET_FPS=15
TEMP_DIR="${TEMP_DIR:-/home/aparnabg/orcd/scratch/entitysam_temp}"

echo "Task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
mkdir -p "$TEMP_DIR"

# --- Run ---
cd /src/sailsprep/tracking_pose_model_testing || exit 1
echo "Running entitysam.py..."
echo "------------------------------------------"

python /src/sailsprep/tracking_pose_model_testing/entitysam.py \
    --csv_file "$CSV_FILE" \
    --ckpt_dir "$CKPT_DIR" \
    --model_cfg "$MODEL_CFG" \
    --mask_decoder_depth $MASK_DECODER_DEPTH \
    --target_fps $TARGET_FPS \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_tasks $SLURM_ARRAY_TASK_COUNT \
    --temp_dir "$TEMP_DIR"

EXIT_CODE=$?
echo "------------------------------------------"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
