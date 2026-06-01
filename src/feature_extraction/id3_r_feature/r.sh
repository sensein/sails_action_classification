#!/bin/bash
#SBATCH --job-name=r2_clip
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --array=0
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/id3_r_feature/logs/%x_%j.log
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/id3_r_feature/logs/%x_%j.err

# ── Usage ──────────────────────────────────────────────────
# sbatch r.sh rmm
# sbatch r.sh loco
# ───────────────────────────────────────────────────────────

TASK="${1}"
if [[ "${TASK}" != "rmm" && "${TASK}" != "loco" ]]; then
    echo "ERROR: must pass task as argument: sbatch job.sh rmm  OR  sbatch job.sh loco"
    exit 1
fi

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate mlasformer

BASE="/home/aparnabg/orcd/scratch/all_project_files/id3_r_feature"
SPLITS_CSV="/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
OUT_ROOT="/orcd/data/satra/002/projects/SAILS/vjepa_features/action_model_outputs/features"
SCRIPT="${BASE}/extract_i3d_r2plus1d_clips.py"

mkdir -p "${OUT_ROOT}"
mkdir -p "${BASE}/logs"

cd "${BASE}"

echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Task      : ${TASK}"
echo "Splits CSV: ${SPLITS_CSV}"
echo "Videos    : $(tail -n +2 ${SPLITS_CSV} | wc -l)"
echo "Start time: $(date)"
echo "=========================================="

NUM_TASKS=1    # must match --array upper bound + 1

python "${SCRIPT}" \
    --splits_csv "${SPLITS_CSV}" \
    --out_root   "${OUT_ROOT}" \
    --task       "${TASK}" \
    --batch_size 8 \
    --gpu        0 \
    --backbone   r2plus1d \
    --task_id    ${SLURM_ARRAY_TASK_ID} \
    --num_tasks  ${NUM_TASKS}  
echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="