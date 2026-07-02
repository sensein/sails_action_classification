#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --job-name=tad_loco
#SBATCH --output=/src/sailsprep/action_model_testing/OpenTAD/logs/tad_%A_%a.out
#SBATCH --error=/src/sailsprep/action_model_testing/OpenTAD/logs/tad_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-8
# ============================================================
#
# Array index mapping:
#   0,1,2  → actionformer × seeds 42,123,456
#   3,4,5  → tridet       × seeds 42,123,456
#   6,7,8  → dyfadet      × seeds 42,123,456
#
# Usage:
#   sbatch submit_tad_array.sh locomotion
#   sbatch submit_tad_array.sh rmm
#
#   # Single model only (e.g. actionformer):
#   sbatch --array=0-2 submit_tad_array.sh locomotion
#
#   # Single experiment (e.g. tridet seed=123):
#   sbatch --array=4 submit_tad_array.sh locomotion
# ============================================================

TASK=${1:?"ERROR: pass task: sbatch submit_tad_array.sh locomotion  OR  rmm"}

if [[ "$TASK" != "locomotion" && "$TASK" != "rmm" ]]; then
    echo "ERROR: TASK must be 'locomotion' or 'rmm', got '${TASK}'"
    exit 1
fi

MODELS=(actionformer tridet dyfadet)
SEEDS=(42 123 456)
BACKBONE=vjepa

NUM_SEEDS=${#SEEDS[@]}
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Unique port per array job to avoid torchrun conflicts
PORT=$((29500 + SLURM_ARRAY_TASK_ID))

# ---- Environment ----
module load miniforge
module load deprecated-modules
module load cuda/11.8.0-x86_64
module load cudnn
conda deactivate
source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
conda activate opentad
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/orcd/pool/007/aparnabg/miniconda3/envs/opentad/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
cd /src/sailsprep/action_model_testing/OpenTAD
mkdir -p logs

echo "LD_LIBRARY_PATH  : ${LD_LIBRARY_PATH}"
echo "nms check        : $(python -c 'import nms_1d_cpu; print("OK")' 2>&1)"
echo "SLURM array task : ${SLURM_ARRAY_TASK_ID}"
echo "Task             : ${TASK}"
echo "Model            : ${MODEL}"
echo "Backbone         : ${BACKBONE}"
echo "Seed             : ${SEED}"
echo "Port             : ${PORT}"
echo "GPU              : ${CUDA_VISIBLE_DEVICES}"
echo "Node             : ${SLURMD_NODENAME}"
echo "Start time       : $(date)"
echo "=========================================="
# Generate config for this seed
python run_locomotion.py \
    --task     ${TASK} \
    --model    ${MODEL} \
    --seed     ${SEED} \
    --mode     generate_config

# Train then test
python run_locomotion.py \
    --task     ${TASK} \
    --model    ${MODEL} \
    --seed     ${SEED} \
    --mode     train_test \
    --gpus     1 \
    --port     ${PORT}
# Generate config for this seed


TRAIN_EXIT=$?

# ---- Frame-level metrics ----
EXP_DIR="exps/${TASK}/${MODEL}_${BACKBONE}/seed_${SEED}"
PRED_JSON=""

if [ -d "${EXP_DIR}" ]; then
    PRED_JSON=$(find ${EXP_DIR} -name "test_results.json" -type f 2>/dev/null | head -1)
fi

if [ -n "${PRED_JSON}" ]; then
    METRICS_DIR="$(dirname ${PRED_JSON})/frame_metrics"
    echo ""
    echo "=========================================="
    echo "Running frame-level metrics..."
    echo "  Predictions : ${PRED_JSON}"
    echo "  Output      : ${METRICS_DIR}"
    echo "=========================================="
    python eval_frame_metrics.py \
        --task       ${TASK} \
        --pred_json  ${PRED_JSON} \
        --output_dir ${METRICS_DIR}
else
    echo "WARNING: No test_results.json found in ${EXP_DIR}"
fi

echo "=========================================="
echo "Done: ${TASK} / ${MODEL} + ${BACKBONE} / seed=${SEED}"
echo "Exit code  : ${TRAIN_EXIT}"
echo "End time   : $(date)"
echo "=========================================="
exit ${TRAIN_EXIT}