#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=${SCRIPT_DIR:-/src/sailsprep/action_model_testing/vjepa/full_video}/logs/%x_%A_%a.out
#SBATCH --error=${SCRIPT_DIR:-/src/sailsprep/action_model_testing/vjepa/full_video}/logs/%x_%A_%a.err

# ── Usage ──────────────────────────────────────────────────────────────
# sbatch --job-name=flat_loco        run_probe.sh locomotion flat
# sbatch --job-name=flat_rmm         run_probe.sh rmm        flat
# sbatch --job-name=hier_loco        run_probe.sh locomotion hierarchical
# sbatch --job-name=hier_rmm         run_probe.sh rmm        hierarchical
# ───────────────────────────────────────────────────────────────────────

TASK=$1
MODE=$2   # flat | hierarchical

if [[ -z "$TASK" || -z "$MODE" ]]; then
    echo "ERROR: Usage: sbatch run_probe.sh <locomotion|rmm> <flat|hierarchical>"
    exit 1
fi

if [[ "$TASK" != "locomotion" && "$TASK" != "rmm" ]]; then
    echo "ERROR: task must be 'locomotion' or 'rmm'"
    exit 1
fi

if [[ "$MODE" != "flat" && "$MODE" != "hierarchical" ]]; then
    echo "ERROR: mode must be 'flat' or 'hierarchical'"
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

if [[ "$MODE" == "flat" ]]; then
    SCRIPT="${SCRIPT_DIR}/train_probe_framelevel.py"
else
    SCRIPT="${SCRIPT_DIR}/train_probe_framelevel_hierarchical.py"
fi

echo "Task=${TASK}  Mode=${MODE}  Seed=${SEED}  array_id=${SLURM_ARRAY_TASK_ID}: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Script: ${SCRIPT}"

python ${SCRIPT} --task ${TASK} --seed ${SEED}

echo "Done task=${TASK} mode=${MODE} seed=${SEED}: $(date)"