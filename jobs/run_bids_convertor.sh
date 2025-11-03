#!/bin/bash
#SBATCH --job-name=bids_processing
#SBATCH --partition=mit_normal
#SBATCH --array=0-19
#SBATCH --output=logs/bids_%A_%a.out
#SBATCH --error=logs/bids_%A_%a.err
#SBATCH --mem=5G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=5

# --- Environment setup ---
# Determine project root dynamically
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)

cd "$PROJECT_ROOT"
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "Job started at $(date) on node $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"

echo "FFmpeg version:"
ffmpeg -version

# Activate poetry env from project root
source $(poetry env info --path)/bin/activate

cd src
echo "Using Python from: $(which python)"
echo "Starting BIDS conversion at $(date)"

# Run your script
python BIDS_convertor.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT

echo "Finished at $(date)"
