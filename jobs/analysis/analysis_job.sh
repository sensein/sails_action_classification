#!/bin/bash
#SBATCH --job-name=stat
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --array=0-8
#SBATCH --output=/src/sailsprep/analysis/logs/job_%A_%a.out
#SBATCH --error=/src/sailsprep/analysis/logs/job_%A_%a.err

mkdir -p /src/sailsprep/analysis/logs

module load miniforge
module load cuda
module load cudnn

conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate analysis

SCRIPTS=(
    "/src/sailsprep/analysis/spinning/spinning.py"
    "/src/sailsprep/analysis/rocking/rocking.py"
    "/src/sailsprep/analysis/jumping/jumping.py"
    "src/sailsprep/analysis/crawling/crawling.py"
    "src/sailsprep/analysis/crusing/crusing.py"
    "src/sailsprep/analysis/handflapping/handflapping.py"
    "src/sailsprep/analysis/loco_combined/loco_combined.py"
    "src/sailsprep/analysis/rmm_combined/rmm_combined.py"
    "src/sailsprep/analysis/running/running.py"
    "src/sailsprep/analysis/walking/walking.py"
)
#add files 
python "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"