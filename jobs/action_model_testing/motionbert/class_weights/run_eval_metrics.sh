#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/logs/eval_metrics_%j.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/logs/eval_metrics_%j.err

mkdir -p /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/logs

module load miniforge
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate motionbert_env

SCRIPT_DIR="/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

EXPERIMENTS=(
    "output_no_weights"
    "output_v3_oversampled"
    "output_weights"
    "output_clips_v3_balanced"
)

for exp in "${EXPERIMENTS[@]}"; do
    PRED_PATH="${SCRIPT_DIR}/${exp}/predictions/action_predictions.json"
    SAVE_DIR="${SCRIPT_DIR}/${exp}/predictions"

    echo ""
    echo "####################################################################"
    echo "  EXPERIMENT: ${exp}"
    echo "####################################################################"

    if [ ! -f "$PRED_PATH" ]; then
        echo "  [SKIP] Predictions not found: $PRED_PATH"
        continue
    fi

    python compute_metrics.py \
        --predictions "$PRED_PATH" \
        --save-dir "$SAVE_DIR"

    echo ""
    echo "  -> Saved to: ${SAVE_DIR}/evaluation_metrics.json"
done

echo ""
echo "=========================================="
echo "All evaluations complete: $(date)"
echo "=========================================="