#!/bin/bash
#SBATCH --job-name=vjepa_extract
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=10:00:00
#SBATCH --output=/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/vjepa/logs/extract_%j.out
#SBATCH --error=/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/vjepa/logs/extract_%j.err

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

# Make log dir just in case
mkdir -p /orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/vjepa/logs

echo "Starting feature extraction: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "${CODE_DIR}/extract_features.py"

echo "Feature extraction done: $(date)"
