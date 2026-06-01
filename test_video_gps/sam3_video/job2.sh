#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G
#SBATCH --time=5:00:00
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

echo "Job $SLURM_JOB_ID  Array $SLURM_ARRAY_TASK_ID  Node $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"

cd /home/aparnabg/orcd/scratch/all_project_files/sam3_video/
python chunks.py \
    --output_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \
    --video_root /orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/derivatives/preprocessed \
    --resize 192 \
    --hybrid_threshold 1500 \
    --overwrite

echo "End: $(date)"