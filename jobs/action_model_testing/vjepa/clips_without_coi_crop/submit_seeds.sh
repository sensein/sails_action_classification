#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=pi_satra 
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=${CODE_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video}/logs/probe_%A_%a.out
#SBATCH --error=${CODE_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video}/logs/probe_%A_%a.err

CODE_DIR="${CODE_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video}"
export HF_HOME="${HF_HOME:-/home/aparnabg/.cache/huggingface}"
export PYTHONUNBUFFERED=1

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate vjepa2-312

# Map array index -> seed
SEEDS=(42 456 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Starting probe training  seed=${SEED}  array_id=${SLURM_ARRAY_TASK_ID}: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "${CODE_DIR}/train_probe.py" \
    --seed ${SEED}

echo "Probe training done  seed=${SEED}: $(date)"
