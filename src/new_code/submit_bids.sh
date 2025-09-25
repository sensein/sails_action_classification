#!/bin/bash
#SBATCH --job-name=bids_processing
#SBATCH --partition=mit_normal
#SBATCH --array=0-50
#SBATCH --output=logs/bids_%A_%a.out
#SBATCH --error=logs/bids_%A_%a.err
#SBATCH --mem=10G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10

mkdir -p logs

module load miniforge


source $(conda info --base)/etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"

conda activate data_env

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

echo "Starting video processing for task $SLURM_ARRAY_TASK_ID"
python /home/aparnabg/bids.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT

echo "Job completed at: $(date)"
