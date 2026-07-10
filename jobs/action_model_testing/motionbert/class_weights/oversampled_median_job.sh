#!/bin/bash
#SBATCH --job-name=motionbert
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/logs/motionbert_%j.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/logs/motionbert_%j.err

mkdir -p /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/logs

module load miniforge
module load cuda
module load cudnn

conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate motionbert_env

which python

export MASTER_CSV="/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v3_oversampled_median.csv"
export OUTPUT_ROOT="/orcd/data/satra/002/projects/SAILS/feature_processing/pipeline_outputs/single_child_videos_motion_bert/"
export ACTION_OUTPUT_ROOT="/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/output_oversampled_median"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "Pose output:   $OUTPUT_ROOT"
echo "Action output: $ACTION_OUTPUT_ROOT"
echo "Master CSV:    $MASTER_CSV"
echo "=========================================="

cd /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/

# Step 1: Run the full pipeline (pose extraction, lifting, finetuning, inference)
python fine_tune_oversample_weights.py --step all --device cuda

# Step 2: Compute evaluation metrics from the predictions
PRED_PATH="${ACTION_OUTPUT_ROOT}/predictions/action_predictions.json"

if [ -f "$PRED_PATH" ]; then
    echo ""
    echo "=========================================="
    echo "Computing evaluation metrics..."
    echo "=========================================="
    python compute_metrics.py \
        --predictions "$PRED_PATH" \
        --save-dir "${ACTION_OUTPUT_ROOT}/predictions/"
else
    echo "[WARN] Predictions file not found at $PRED_PATH, skipping metrics."
fi

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="