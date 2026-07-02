#!/bin/bash
#SBATCH --job-name=id3_r2_full
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=/src/sailsprep/action_model_testing/feature_extraction/logs/%x_%j.log
#SBATCH --error=/src/sailsprep/action_model_testing/feature_extraction/logs/%x_%j.err
#SBATCH --array=0-7
#SBATCH --mem=64G

# ── Usage ──────────────────────────────────────────────────
# sbatch jobs/action_model_testing/i3d_r2p1d_feature_extracion.sh i3d
# sbatch jobs/action_model_testing/i3d_r2p1d_feature_extracion.sh r2plus1d
# ───────────────────────────────────────────────────────────

BACKBONE="${1}"
if [[ "${BACKBONE}" != "i3d" && "${BACKBONE}" != "r2plus1d" ]]; then
    echo "ERROR: must pass backbone as argument:"
    echo "  sbatch full_job.sh i3d"
    echo "  sbatch full_job.sh r2plus1d"
    exit 1
fi

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate mlasformer

BASE="/src/sailsprep/action_model_testing/feature_extraction"
SPLITS_CSV="/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
OUT_ROOT="/orcd/data/satra/002/projects/SAILS/action_outputs_features/feature_dir/features_full_video_interpolated_h5_i3d_r2"
SCRIPT="${BASE}/i3d_extractor.py"

mkdir -p "${OUT_ROOT}"
mkdir -p "${BASE}/logs"

cd "${BASE}"

echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Backbone  : ${BACKBONE}"
echo "Splits CSV: ${SPLITS_CSV}"
echo "Videos    : $(tail -n +2 ${SPLITS_CSV} | wc -l)"
echo "Start time: $(date)"
echo "=========================================="

NUM_TASKS=8

python "${SCRIPT}" \
    --splits_csv "${SPLITS_CSV}" \
    --output_dir "${OUT_ROOT}" \
    --backbone   "${BACKBONE}" \
    --batch_size 8 \
    --gpu        0 \
    --task_id    ${SLURM_ARRAY_TASK_ID} \
    --num_tasks  ${NUM_TASKS}

echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="