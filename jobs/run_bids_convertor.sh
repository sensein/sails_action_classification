#!/bin/bash
#SBATCH --job-name=bids_processing
#SBATCH --partition=mit_normal
#SBATCH --array=0-19
#SBATCH --output=logs/bids_%A_%a.out
#SBATCH --error=logs/bids_%A_%a.err
#SBATCH --mem=5G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=5

mkdir -p logs

# --- Determine project root robustly ---
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR" || { echo "❌ Cannot cd to SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"; exit 1; }
else
    SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    cd "$SCRIPT_DIR/.." || { echo "❌ Cannot cd to project root"; exit 1; }
fi

echo "Running from project root: $(pwd)"
export PYTHONUNBUFFERED=1

ffmpeg -version || echo "⚠️ FFmpeg not available"

# --- Poetry setup ---
if ! poetry env info --path &> /dev/null; then
    echo "Creating Poetry environment..."
    poetry install || { echo "❌ Poetry install failed"; exit 1; }
fi

ENV_PATH=$(poetry env info --path)
source "$ENV_PATH/bin/activate" || { echo "❌ Failed to activate Poetry environment"; exit 1; }

echo "Using Python from: $(which python)"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Starting BIDS conversion at $(date)"

python -m sailsprep.BIDS_convertor "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_MAX"

echo "Finished at $(date)"
