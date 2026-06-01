#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=1200G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/sam3_video/logs/bbox_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/sam3_video/logs/bbox_%A_%a.err

mkdir -p /home/aparnabg/orcd/scratch/all_project_files/sam3_video/logs
module load miniforge
module load cuda
module load cudnn
conda deactivate
source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
conda activate hf_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Array Task  : $SLURM_ARRAY_TASK_ID / 5"
echo "Partition   : $SLURM_JOB_PARTITION"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time  : $(date)"
echo "=========================================="

cd /home/aparnabg/orcd/scratch/all_project_files/sam3_video/
python job2.py \
    --csv  /home/aparnabg/orcd/scratch/all_project_files/muti_lable_model_testing/splits_clean_ViT_H.csv \
    --output_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \
    --multi \
    --hybrid_threshold 1500 \
    --resize 360 \
    --resize_long 320 \
    --resize_xlong 256 \
    --long_threshold 1500 \
    --xlong_threshold 3000 \
    --empty_cache_every 50 \
    --rebuild_backend_every 25 \
    --array_index $SLURM_ARRAY_TASK_ID \
    --array_total 1

echo "End time: $(date)"