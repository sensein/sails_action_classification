#!/bin/bash
#SBATCH --job-name=window_vlm
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/2sec_window/logs/%x_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/2sec_window/logs/%x_%A_%a.err

# ============================================================
# Unified 2-sec window VLM classifier — Ovis2 or Qwen2.5-VL
#
# Usage (deterministic):
#   sbatch --job-name=ovis_a_loco  submit_window.sh ovis a loco
#   sbatch --job-name=qwen_b_rmm   submit_window.sh qwen b rmm
#
# Usage (random frame sampling for spread metrics):
#   sbatch --job-name=ovis_a_loco_s42  submit_window.sh ovis a loco 42
#   sbatch --job-name=ovis_a_loco_s123 submit_window.sh ovis a loco 123
#   sbatch --job-name=qwen_a_rmm_s42   submit_window.sh qwen a rmm 42
#
# Override array size:
#   sbatch --array=0-1 submit_window.sh ovis b loco 42
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Parse and validate arguments
# ---------------------------------------------------------------------------
MODEL_KEY=${1:?"ERROR: arg 1 must be 'ovis' or 'qwen'"}
APPROACH=${2:?"ERROR: arg 2 must be 'a', 'b', or 'c'"}
TASK=${3:?"ERROR: arg 3 must be 'loco' or 'rmm'"}
SEED=${4:-""}   # Optional. If provided, enables --random-frames.

if [[ "${MODEL_KEY}" != "ovis" && "${MODEL_KEY}" != "qwen" ]]; then
    echo "ERROR: MODEL must be 'ovis' or 'qwen', got '${MODEL_KEY}'"
    exit 1
fi
if [[ "${APPROACH}" != "a" && "${APPROACH}" != "b" && "${APPROACH}" != "c" ]]; then
    echo "ERROR: APPROACH must be 'a', 'b', or 'c', got '${APPROACH}'"
    exit 1
fi
if [[ "${TASK}" != "loco" && "${TASK}" != "rmm" ]]; then
    echo "ERROR: TASK must be 'loco' or 'rmm', got '${TASK}'"
    exit 1
fi

# Build optional random-frame flags.
RANDOM_FRAME_ARGS=""
if [[ -n "${SEED}" ]]; then
    RANDOM_FRAME_ARGS="--random-frames --seed ${SEED}"
fi

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/2sec_window"
CSV_FILE="/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
export HF_HOME="/orcd/data/satra/002/huggingface"
export TRANSFORMERS_CACHE="/orcd/data/satra/002/huggingface"
export HF_DATASETS_CACHE="/orcd/data/satra/002/huggingface"

BASE_OUTPUT="/orcd/data/satra/002/projects/SAILS/vjepa_features/action_model_outputs/vlm_models/p_v1"

if [[ "${MODEL_KEY}" == "ovis" ]]; then
    SCRIPT_NAME="window_classifier_ovis.py"
    BASE_MODEL_DIR="${BASE_OUTPUT}/ovis/window_${TASK}/approach_${APPROACH}"
else
    SCRIPT_NAME="window_classifier_qwen.py"
    BASE_MODEL_DIR="${BASE_OUTPUT}/qwen2_5/window_${TASK}/approach_${APPROACH}"
fi

# Seed-specific subdir so runs never overwrite each other.
if [[ -n "${SEED}" ]]; then
    OUTPUT_DIR="${BASE_MODEL_DIR}/seed_${SEED}"
else
    OUTPUT_DIR="${BASE_MODEL_DIR}"
fi

# ---------------------------------------------------------------------------
# 3. Environment — model-specific conda envs
#    Ovis  → ovis env      from /home/aparnabg/orcd/scratch/miniconda3
#    Qwen  → ovis_qwen_env from /home/aparnabg/orcd/pool/miniconda3
# ---------------------------------------------------------------------------
module purge
module load miniforge
module load cuda
module load cudnn

conda deactivate 2>/dev/null || true

if [[ "${MODEL_KEY}" == "ovis" ]]; then
    source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
    conda activate ovis
else
    source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
    conda activate ovis_qwen_env
fi

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

TEMP_CSV="${OUTPUT_DIR}/temp_${MODEL_KEY}_${APPROACH}_${TASK}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
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
echo "Model         : ${MODEL_KEY}"
echo "Approach      : ${APPROACH}"
echo "Task          : ${TASK}"
echo "CSV           : ${CSV_FILE}"
echo "Chunk rows    : ${CHUNK_ROWS}  (lines ${START_LINE}–${END_LINE})"
echo "Output dir    : ${OUTPUT_DIR}"
echo "GPU           : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Random frames : ${RANDOM_FRAME_ARGS:-no (deterministic linspace)}"
echo "Conda env     : ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python        : $(which python)"
echo "Start time    : $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 7. Build and run the python command
# ---------------------------------------------------------------------------
CMD="python3 ${SCRIPT_DIR}/${SCRIPT_NAME} \
    --task ${TASK} \
    --approach ${APPROACH} \
    --csv ${TEMP_CSV} \
    --video-col video_path \
    --label-col label_path \
    --output-dir ${OUTPUT_DIR} \
    --num-frames 6 \
    ${RANDOM_FRAME_ARGS}"

if [[ "${MODEL_KEY}" == "ovis" ]]; then
    CMD="${CMD} --model AIDC-AI/Ovis2-8B --max-partition 9 --no-flash-attn"
else
    CMD="${CMD} --model Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16"
fi

echo "Command: ${CMD}"
echo "=========================================="

eval ${CMD}
EXIT_CODE=$?

# ---------------------------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------------------------
rm -f "${TEMP_CSV}"

echo "=========================================="
echo "Model         : ${MODEL_KEY}"
echo "Approach      : ${APPROACH}"
echo "Task          : ${TASK}"
echo "Array task    : ${SLURM_ARRAY_TASK_ID}"
echo "Seed          : ${SEED:-N/A}"
echo "Exit code     : ${EXIT_CODE}"
echo "End time      : $(date)"
echo "=========================================="

exit ${EXIT_CODE}