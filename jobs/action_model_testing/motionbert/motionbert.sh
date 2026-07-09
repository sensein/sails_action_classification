#!/bin/bash
#SBATCH --job-name=motionbert
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/src/sailsprep/action_model_testing/motionbert/logs/motionbert_%j.out
#SBATCH --error=/src/sailsprep/action_model_testing/motionbert/logs/motionbert_%j.err

mkdir -p /src/sailsprep/action_model_testing/motionbert/logs

module load miniforge
module load cuda
module load cudnn
conda deactivate
CONDA_SH="${CONDA_SH:-/home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate motionbert_env



export MASTER_CSV="${MASTER_CSV:-/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v2.csv}"
export OUTPUT_ROOT="/orcd/data/satra/002/projects/SAILS/action_outputs_features/action_model_outputs/single_child_videos_motion_bert/"
export ACTION_OUTPUT_ROOT="${ACTION_OUTPUT_ROOT:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/motionbert/output_no_weights}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "Pose output:   $OUTPUT_ROOT"
echo "Action output: $ACTION_OUTPUT_ROOT"
echo "Master CSV:    $MASTER_CSV"
echo "=========================================="

cd /src/sailsprep/action_model_testing/motionbert/

python motionbert.py --step all --device cuda

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="