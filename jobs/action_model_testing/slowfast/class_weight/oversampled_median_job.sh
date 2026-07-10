#!/bin/bash
#SBATCH --job-name=slowfast_finetune
#SBATCH --partition=ou_bcs_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/class_weight/logs/slowfast_median_%j.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/class_weight/logs/slowfas_mediant_%j.err


mkdir -p /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/class_weight/logs

module load miniforge
module load cuda
module load cudnn
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate pytorchvideo_env
export MASTER_CSV="/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v3_oversampled_median.csv"
export OUTPUT_DIR="/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/class_weight/output_v3_oversampled_median/"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "MASTER_CSV: $MASTER_CSV"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "=========================================="

cd /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/class_weight/
python slowfast_finetune.py

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="