#!/bin/bash
#SBATCH --job-name=qwen_clip
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/clips/logs/qwen_%x_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/clips/logs/qwen_%x_%A_%a.err

# ============================================================
# Unified Qwen2.5-VL clip-level classifier — locomotion or RMM.
#
# Usage (deterministic, original behaviour):
#   sbatch submit_qwen_clip.sh loco
#   sbatch submit_qwen_clip.sh rmm
#
# Usage (random frame sampling — vary SEED to measure metric spread):
#   sbatch submit_qwen_clip.sh loco 42
#   sbatch submit_qwen_clip.sh loco 123
#   sbatch submit_qwen_clip.sh loco 999
#
# Override array size:
#   sbatch --array=0-2 submit_qwen_clip.sh loco 42
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Validate arguments
# ---------------------------------------------------------------------------
TASK=${1:?"ERROR: pass task as first argument — e.g.  sbatch submit_qwen_clip.sh loco  OR  rmm"}
SEED=${2:-""}   # Optional. If provided, enables --random-frames.

if [[ "${TASK}" != "loco" && "${TASK}" != "rmm" ]]; then
    echo "ERROR: TASK must be 'loco' or 'rmm', got '${TASK}'"
    exit 1
fi

# Build the optional random-frame flags passed to the Python script.
RANDOM_FRAME_ARGS=""
if [[ -n "${SEED}" ]]; then
    RANDOM_FRAME_ARGS="--random-frames --seed ${SEED}"
fi

# ---------------------------------------------------------------------------
# 2. Per-task paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/clips"

if [[ "${TASK}" == "loco" ]]; then
    CSV_FILE="/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v2.csv"
    BASE_OUTPUT="/orcd/data/satra/002/projects/SAILS/vjepa_features/action_model_outputs/vlm_models/qwen/clips_loco"
else
    CSV_FILE="/home/aparnabg/orcd/scratch/all_project_files/splits_rmm_cut-clips_v1.csv"
    BASE_OUTPUT="/orcd/data/satra/002/projects/SAILS/vjepa_features/action_model_outputs/vlm_models/qwen/clips_rmm"
fi

# Seed-specific subdirectory so runs never overwrite each other.
if [[ -n "${SEED}" ]]; then
    OUTPUT_DIR="${BASE_OUTPUT}/seed_${SEED}"
else
    OUTPUT_DIR="${BASE_OUTPUT}"
fi

# ---------------------------------------------------------------------------
# 3. Environment
# ---------------------------------------------------------------------------

module load miniforge
module load cuda
module load cudnn

conda deactivate 
source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
conda activate ovis_qwen_env

export HF_HOME="/orcd/data/satra/002/huggingface"
export TRANSFORMERS_CACHE="/orcd/data/satra/002/huggingface"
export HF_DATASETS_CACHE="/orcd/data/satra/002/huggingface"

# ---------------------------------------------------------------------------
# 4. Directories
# ---------------------------------------------------------------------------
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 5. Chunk the CSV for this array job
# ---------------------------------------------------------------------------
TOTAL_VIDEOS=$(tail -n +2 "${CSV_FILE}" | wc -l)
VIDEOS_PER_JOB=$(( (TOTAL_VIDEOS + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
START_LINE=$(( SLURM_ARRAY_TASK_ID * VIDEOS_PER_JOB + 2 ))
END_LINE=$(( START_LINE + VIDEOS_PER_JOB - 1 ))

if [[ ${END_LINE} -gt $(( TOTAL_VIDEOS + 1 )) ]]; then
    END_LINE=$(( TOTAL_VIDEOS + 1 ))
fi

TEMP_CSV="${OUTPUT_DIR}/temp_${TASK}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
head -n 1 "${CSV_FILE}" > "${TEMP_CSV}"
sed -n "${START_LINE},${END_LINE}p" "${CSV_FILE}" >> "${TEMP_CSV}"

CHUNK_ROWS=$(( END_LINE - START_LINE + 1 ))

# ---------------------------------------------------------------------------
# 6. Log header
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Array task    : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
echo "Node          : ${SLURMD_NODENAME}"
echo "Task          : ${TASK}"
echo "CSV           : ${CSV_FILE}"
echo "Chunk rows    : ${CHUNK_ROWS}  (lines ${START_LINE}–${END_LINE})"
echo "Temp CSV      : ${TEMP_CSV}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "GPU           : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Random frames : ${RANDOM_FRAME_ARGS:-no (deterministic linspace)}"
echo "Python        : $(which python)"
echo "Start time    : $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 7. Run classifier
# ---------------------------------------------------------------------------
python3 "${SCRIPT_DIR}/qwen_clip_classifier.py" \
    --task "${TASK}" \
    --csv "${TEMP_CSV}" \
    --clip-column cut_clip_path \
    --output-dir "${OUTPUT_DIR}" \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --num-frames 8 \
    --dtype bfloat16 \
    ${RANDOM_FRAME_ARGS}

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------------------------
rm -f "${TEMP_CSV}"

echo "=========================================="
echo "Task          : ${TASK}"
echo "Array task    : ${SLURM_ARRAY_TASK_ID}"
echo "Seed          : ${SEED:-N/A}"
echo "Exit code     : ${EXIT_CODE}"
echo "End time      : $(date)"
echo "=========================================="

exit ${EXIT_CODE}