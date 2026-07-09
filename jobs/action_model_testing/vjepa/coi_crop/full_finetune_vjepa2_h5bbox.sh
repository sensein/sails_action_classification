#!/bin/bash
#SBATCH --job-name=vjepa_ft
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --output=${CODE_DIR:-/src/sailsprep/action_model_testing/vjepa/coi_crop}/logs/vjepa_ft_%j.out
#SBATCH --error=${CODE_DIR:-/src/sailsprep/action_model_testing/vjepa/coi_crop}/logs/vjepa_ft_%j.err

CODE_DIR="${CODE_DIR:-/src/sailsprep/action_model_testing/vjepa/coi_crop}"
mkdir -p "${CODE_DIR}/logs"

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate vjepa2-312

echo "Job $SLURM_JOB_ID  $(date)"
cd "${CODE_DIR}"

python finetune_vjepa2_h5bbox.py --full_finetune

echo "End $(date)"