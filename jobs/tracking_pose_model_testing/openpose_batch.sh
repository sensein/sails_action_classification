#!/bin/bash
#SBATCH --job-name=openpose_batch
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=openpose_%j.out
#SBATCH --error=openpose_%j.err

module load miniforge
module load cudnn
module load cuda
module load apptainer
conda deactivate

CSV_PATH="/home/aparnabg/orcd/scratch/csv2_filtered_multiple_people.csv"
OUTPUT_DIR="/home/aparnabg/orcd/scratch/openpose_output"
MODEL_FOLDER="/home/aparnabg/orcd/scratch/openpose/models"
SCRIPT_PATH="/home/aparnabg/orcd/scratch/openpose_video.py"

singularity exec --nv \
  --bind /orcd/scratch:/orcd/scratch \
  --bind /home/aparnabg/orcd/scratch:/home/aparnabg/orcd/scratch \
  --bind /home/aparnabg/orcd/scratch/ffmpeg-7.0.2-amd64-static/ffmpeg:/usr/bin/ffmpeg \
  /home/aparnabg/orcd/scratch/openpose-final.sif \
  python3 "$SCRIPT_PATH" "$CSV_PATH" "$OUTPUT_DIR" "$MODEL_FOLDER"

echo "Job completed"