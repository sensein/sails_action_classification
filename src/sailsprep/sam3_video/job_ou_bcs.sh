#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=2:00:00
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/sam3_video/logs/bbox_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/sam3_video/logs/bbox_%A_%a.err

mkdir -p /home/aparnabg/orcd/scratch/all_project_files/sam3_video/logs
module load miniforge
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate app_env


cd /home/aparnabg/orcd/scratch/all_project_files/sam3_video/
python generate_masked_videos.py \
    --results_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \
    --multi

echo "End time: $(date)"