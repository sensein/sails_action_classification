#!/bin/bash
#SBATCH --job-name=vjepa_full
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=10:00:00
#SBATCH --array=0
#SBATCH --output=/src/sailsprep/action_model_testing/feature_extraction/logs/vjepa_%A_%a.out
#SBATCH --error=/src/sailsprep/action_model_testing/feature_extraction/logs/vjepa_%A_%a.err

mkdir -p /src/sailsprep/action_model_testing/feature_extraction/logs
export HF_HOME="${HF_HOME:-/home/aparnabg/.cache/huggingface}"
export PYTHONUNBUFFERED=1

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate vjepa2-312

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)  Start: $(date)"
echo "=========================================="

SPLITS_CSV="${SPLITS_CSV:-/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv}"

cd /src/sailsprep/action_model_testing/feature_extraction
python -u vjepa2_extractor.py \
    --splits_csv "${SPLITS_CSV}" \
    --output_dir /orcd/data/satra/002/projects/SAILS/action_outputs_features/feature_dir/vjepa_features_h5/full_video_introp_h5_vjepa_features \
    --target_fps 15 \
    --crop_size 256 \
    --batch_clips 2 \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_tasks 1
echo "End: $(date)"