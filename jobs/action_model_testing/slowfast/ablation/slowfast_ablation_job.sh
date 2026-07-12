#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=2-10
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/logs/ablation_v%a_%j.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/logs/ablation_v%a_%j.err

mkdir -p /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/logs

module load miniforge
module load cuda
module load cudnn

conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate pytorchvideo_env

cd /src/sailsprep/action_model_testing/slow_fast/experiments/ablation

# SLURM_ARRAY_TASK_ID will be 2, 3, 4, ..., 10
VERSION="v${SLURM_ARRAY_TASK_ID}"

echo "============================================"
echo "  Running ablation version: ${VERSION}"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

python train.py --version ${VERSION}

echo "Done with ${VERSION}"