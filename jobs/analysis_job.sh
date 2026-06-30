#!/bin/bash
#SBATCH --job-name=stat
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/logs/job_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/logs/job_%A_%a.err

mkdir -p /home/aparnabg/orcd/scratch/all_project_files/logs

module load miniforge
module load cuda
module load cudnn

conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate app_env

SCRIPTS=(
    "/home/aparnabg/orcd/scratch/all_project_files/spinning/spinning.py",
    "/home/aparnabg/orcd/scratch/all_project_files/rocking/rocking.py",
    "/home/aparnabg/orcd/scratch/all_project_files/jumping/jumping.py",
)

python "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"