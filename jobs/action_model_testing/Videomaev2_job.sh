#!/bin/bash
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/Videomaev2/clip/logs/videomae_%j_%x.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/Videomaev2/clip/logs/videomae_%j_%x.err

# ============================================================
# Usage (called by submit_all.sh, not directly):
#   sbatch --job-name=vmae_clip_loco_42    job.sh clip    loco 42
#   sbatch --job-name=vmae_fullvid_rmm_123 job.sh fullvid loco  123
#   sbatch --job-name=vmae_twostage_loco_456 job.sh twostage loco 456
# ============================================================

MODE=${1:-clip}    # clip | fullvid | twostage
LABEL=${2:-loco}   # loco | rmm
SEED=${3:-42}

# Validate
if [[ "$LABEL" != "loco" && "$LABEL" != "rmm" ]]; then
    echo "ERROR: LABEL must be 'loco' or 'rmm', got: $LABEL"; exit 1
fi
if [[ "$MODE" != "clip" && "$MODE" != "fullvid" && "$MODE" != "twostage" ]]; then
    echo "ERROR: MODE must be 'clip', 'fullvid', or 'twostage', got: $MODE"; exit 1
fi

case "$MODE" in
    clip)      SCRIPT="videomae2_finetune.py" ;;
    fullvid)   SCRIPT="videomae2_fullvideo_sliding.py" ;;
    twostage)  SCRIPT="videomae2_twostage_sliding.py" ;;
esac

SCRIPT_DIR=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/Videomaev2/clip
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
echo "Mode      : $MODE  →  $SCRIPT"
echo "Label     : $LABEL"
echo "Seed      : $SEED"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Start time: $(date)"
echo "=========================================="

cd ${SCRIPT_DIR}
python ${SCRIPT} --task ${LABEL} --seed ${SEED}

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="