#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video/rmm/logs/probe_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video/rmm/logs/probe_%A_%a.err

export HF_HOME=/home/aparnabg/.cache/huggingface
export PYTHONUNBUFFERED=1

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312

# Map array index -> seed
SEEDS=(42 456 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Starting RMM probe training  seed=${SEED}  array_id=${SLURM_ARRAY_TASK_ID}: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video/rmm/train_probe.py \
    --seed ${SEED}

echo "RMM probe training done  seed=${SEED}: $(date)"
