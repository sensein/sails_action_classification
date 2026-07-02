#!/bin/bash
#SBATCH --job-name=face_mediapipe
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/logs/face_mediapipe_%j.out
#SBATCH --error=/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/logs/face_mediapipe_%j.err

cd /src/sailsprep/tracking_pose_model_testing
mkdir -p /home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/logs

module load miniforge
module load cuda
module load cudnn
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate pose_model_test

echo "Running face_mediapipe.py"
python /src/sailsprep/tracking_pose_model_testing/face_mediapipe.py
