#!/bin/bash
#SBATCH --job-name=vjepa2_ablation
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=120GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=${SCRIPT_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjepa_crop}/logs/%x_%A.out
#SBATCH --error=${SCRIPT_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjepa_crop}/logs/%x_%A.err

SCRIPT_DIR="${SCRIPT_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjepa_crop}"
mkdir -p "${SCRIPT_DIR}/logs"

module load miniforge/24.3.0-0
module load cuda
module load cudnn
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate vjepa2-312
cd "${SCRIPT_DIR}"

LABEL=$1
SEED=$2

if [ -z "$LABEL" ] || [ -z "$SEED" ]; then
  echo "Error: Usage: sbatch run.sh <label> <seed>"
  exit 1
fi

echo "============================================"
echo "  Label : $LABEL"
echo "  Seed  : $SEED"
echo "============================================"

START_TIME=$(date +%s)
echo "Start time: $(date)"

# Where old extracted features already live (from original run)
OLD_CACHE_BASE="/orcd/data/satra/002/projects/SAILS/vjepa_features/action_model_outputs/clips_h5/vjepa_new_crop/clip_level_ablation"

# Where new seed outputs will go
NEW_BASE="/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vjepa_new_crop/clip_level_ablation"

CACHE_SRC="${OLD_CACHE_BASE}/${LABEL}/extracted_features.pt"
SEED_OUT_DIR="${NEW_BASE}/${LABEL}/seed_${SEED}"
CACHE_DST="${SEED_OUT_DIR}/extracted_features.pt"

# Verify old cache exists
if [ ! -f "$CACHE_SRC" ]; then
  echo "ERROR: old feature cache not found at:"
  echo "       $CACHE_SRC"
  exit 1
fi

# Create output dir and symlink the cache
mkdir -p "$SEED_OUT_DIR"
ln -sf "$CACHE_SRC" "$CACHE_DST"
echo "Symlinked cache: $CACHE_SRC -> $CACHE_DST"

echo "Running seed $SEED with --skip_extraction"
python vjepa_clip_level_ablation.py \
  --label $LABEL \
  --head all \
  --seed $SEED \
  --skip_extraction

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "End time: $(date)"
echo "Total runtime: ${DURATION} seconds ($(($DURATION / 60)) minutes)"