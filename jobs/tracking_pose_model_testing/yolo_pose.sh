#!/bin/bash
#SBATCH --job-name=yolo_pose
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=${POSE_LOG_DIR:-/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/logs}/yolo_pose_%j.out
#SBATCH --error=${POSE_LOG_DIR:-/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/logs}/yolo_pose_%j.err

POSE_LOG_DIR="${POSE_LOG_DIR:-/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/logs}"
cd /src/sailsprep/tracking_pose_model_testing
mkdir -p "${POSE_LOG_DIR}"

module load miniforge
module load cuda
module load cudnn
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate pose_model_test

echo "Running yolo_pose.py"
python /src/sailsprep/tracking_pose_model_testing/yolo_pose.py
