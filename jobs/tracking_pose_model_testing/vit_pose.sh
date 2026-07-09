#!/bin/bash
#SBATCH --job-name=h5_pose
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=${POSE_LOG_DIR:-/home/aparnabg/orcd/scratch/all_project_files/pose_data/logs}/mmpose_%A_%a.out
#SBATCH --error=${POSE_LOG_DIR:-/home/aparnabg/orcd/scratch/all_project_files/pose_data/logs}/mmpose_%A_%a.err

POSE_LOG_DIR="${POSE_LOG_DIR:-/home/aparnabg/orcd/scratch/all_project_files/pose_data/logs}"
MMPOSE_DIR="${MMPOSE_DIR:-/home/aparnabg/orcd/scratch/mmpose}"
SPLIT_CSV="${SPLIT_CSV:-/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv}"
VITPOSE_OUTPUT_DIR="${VITPOSE_OUTPUT_DIR:-/orcd/scratch/bcs/001/sensein/sails/vit_pose_annotaed_full_video/}"

cd "${MMPOSE_DIR}"
mkdir -p "${POSE_LOG_DIR}"

module load miniforge
module load cuda
module load cudnn
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate vitpose_env39

which python

# SLURM_ARRAY_TASK_ID  : 0 or 1  (set automatically by SLURM)
# --num_jobs must match the number of array tasks  (0-1  =>  2 jobs)
python /src/sailsprep/tracking_pose_model_testing/vit_pose.py \
    --split_csv  "${SPLIT_CSV}" \
    --h5_dir     /orcd/data/satra/002/projects/SAILS/vjepa_features/h5_action_rmm_interpolated/h5folders/ \
    --output_dir "${VITPOSE_OUTPUT_DIR}" \
    --model_name usyd-community/vitpose-plus-huge \
    --array_index $SLURM_ARRAY_TASK_ID \
    --num_jobs    5
    