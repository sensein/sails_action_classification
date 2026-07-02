#!/bin/bash
#SBATCH --job-name=mstcn2
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/src/sailsprep/action_model_testing/mstcn2/logs/mstcn2_%j.log
#SBATCH --error=/src/sailsprep/action_model_testing/mstcn2/logs/mstcn2_%j.err

# ============================================================
# USAGE:
#   # Train all 3 seeds + aggregate:
#   sbatch --export=LABEL=loco,FEATURE=i3d mstcn2.sh
#
#   # Predict only (single seed):
#   sbatch --export=LABEL=loco,FEATURE=i3d,ACTION=predict,SEED=42 mstcn2.sh
#
#   # Aggregate existing seed results only:
#   sbatch --export=LABEL=loco,FEATURE=i3d,ACTION=evaluate mstcn2.sh
#
# LABEL   : loco | rmm
# FEATURE : i3d | vjepa | r2plus1d
# ACTION  : train | predict | evaluate   (default: train)
# SEED    : single seed override (default: runs all 3 seeds for train)
# ============================================================

# --- Defaults ---
LABEL=${LABEL:-loco}
FEATURE=${FEATURE:-i3d}
ACTION=${ACTION:-train}
SEED=${SEED:-""}

# --- Validate ---
if [[ "$LABEL" != "loco" && "$LABEL" != "rmm" ]]; then
    echo "ERROR: LABEL must be 'loco' or 'rmm'. Got: '$LABEL'"
    exit 1
fi

if [[ "$FEATURE" != "i3d" && "$FEATURE" != "vjepa" && "$FEATURE" != "r2plus1d" ]]; then
    echo "ERROR: FEATURE must be 'i3d', 'vjepa', or 'r2plus1d'. Got: '$FEATURE'"
    exit 1
fi

if [[ "$ACTION" != "train" && "$ACTION" != "predict" && "$ACTION" != "evaluate" ]]; then
    echo "ERROR: ACTION must be 'train', 'predict', or 'evaluate'. Got: '$ACTION'"
    exit 1
fi

# --- Environment ---
module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate mlasformer
export PYTHONNOUSERSITE=1
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export CUDA_VISIBLE_DEVICES=0

# --- Paths ---
SCRIPT_DIR=/src/sailsprep/action_model_testing/mstcn2
LOG_DIR=${SCRIPT_DIR}/logs
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time   : $(date)"
echo "------------------------------------------"
echo "LABEL        : ${LABEL}"
echo "FEATURE TYPE : ${FEATURE}"
echo "ACTION       : ${ACTION}"
echo "SEED         : ${SEED:-'all (42,123,456)'}"
echo "=========================================="

cd ${SCRIPT_DIR}

# ============================================================
# If ACTION=evaluate, just aggregate existing results
# ============================================================
if [[ "$ACTION" == "evaluate" ]]; then
    echo ">>> Aggregating seed results..."
    python mstcn2.py \
        --label        ${LABEL}   \
        --feature_type ${FEATURE} \
        --action       evaluate   \
        --seed         42
    EXIT_CODE=$?
    echo "Exit code: ${EXIT_CODE}"
    echo "End time : $(date)"
    exit ${EXIT_CODE}
fi

# ============================================================
# If a single SEED is specified, run only that seed
# ============================================================
if [[ -n "$SEED" ]]; then
    echo ">>> Running single seed: ${SEED}"
    python mstcn2.py \
        --label        ${LABEL}   \
        --feature_type ${FEATURE} \
        --action       ${ACTION}  \
        --seed         ${SEED}
    EXIT_CODE=$?
    echo "Exit code: ${EXIT_CODE}"
    echo "End time : $(date)"
    exit ${EXIT_CODE}
fi

# ============================================================
# Default: train with all 3 seeds, then aggregate
# ============================================================
SEEDS=(42 123 456)
FINAL_EXIT=0

for S in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  SEED ${S} — START  $(date)"
    echo "=========================================="

    python mstcn2.py \
        --label        ${LABEL}   \
        --feature_type ${FEATURE} \
        --action       train      \
        --seed         ${S}

    EC=$?
    echo "  SEED ${S} — EXIT CODE: ${EC}"

    if [[ $EC -ne 0 ]]; then
        echo "  WARNING: seed ${S} failed!"
        FINAL_EXIT=$EC
    fi
done

# Aggregate across seeds
echo ""
echo "=========================================="
echo "  AGGREGATING SEED RESULTS"
echo "=========================================="

python mstcn2.py \
    --label        ${LABEL}   \
    --feature_type ${FEATURE} \
    --action       evaluate   \
    --seed         42

echo "=========================================="
echo "All done. Exit code: ${FINAL_EXIT}"
echo "End time : $(date)"
echo "=========================================="

exit ${FINAL_EXIT}