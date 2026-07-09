#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=${SCRIPT_DIR:-/src/sailsprep/action_model_testing/vjepa/full_video}/logs/%x_%A_%a_two.out
#SBATCH --error=${SCRIPT_DIR:-/src/sailsprep/action_model_testing/vjepa/full_video}/logs/%x_%A_%a_two.err

# ── Usage ──────────────────────────────────────────────────
# sbatch --job-name=locomotion two.sh locomotion
# sbatch --job-name=rmm        two.sh rmm
# ───────────────────────────────────────────────────────────

TASK=$1

if [[ -z "$TASK" ]]; then
    echo "ERROR: No task provided. Usage: sbatch two.sh <locomotion|rmm>"
    exit 1
fi

if [[ "$TASK" != "locomotion" && "$TASK" != "rmm" ]]; then
    echo "ERROR: task must be 'locomotion' or 'rmm', got: $TASK"
    exit 1
fi

export HF_HOME="${HF_HOME:-/home/aparnabg/.cache/huggingface}"
export PYTHONUNBUFFERED=1

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate vjepa2-312

SEEDS=(42 456 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SCRIPT_DIR="${SCRIPT_DIR:-/src/sailsprep/action_model_testing/vjepa/full_video}"

echo "Task: ${TASK}  Seed=${SEED}  array_id=${SLURM_ARRAY_TASK_ID}: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "${SCRIPT_DIR}/two_stage.py" \
    --task ${TASK} \
    --seed ${SEED}

echo "Done task=${TASK} seed=${SEED}: $(date)"