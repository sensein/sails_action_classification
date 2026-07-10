#!/bin/bash
#SBATCH --job-name=slowfast_finetune
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/logs/slowfast_%j.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/logs/slowfast_%j.err


mkdir -p /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/logs

module load miniforge
module load cuda
module load cudnn
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate pytorchvideo_env

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Start time: $(date)"
echo "=========================================="

cd /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/
python slowfast_finetune_clip.py
