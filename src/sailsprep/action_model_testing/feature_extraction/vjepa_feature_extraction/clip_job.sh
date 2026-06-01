#!/bin/bash
#SBATCH --job-name=vjepa_clips
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/vjepa_feature_extraction/logs/vjepa_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/vjepa_feature_extraction/logs/vjepa_%A_%a.err

mkdir -p /home/aparnabg/orcd/scratch/all_project_files/vjepa_feature_extraction/logs

module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312

TASK="${1:-rmm}"
if [[ "$TASK" != "rmm" && "$TASK" != "loco" ]]; then
    echo "Usage: sbatch job.sh [rmm|loco]"; exit 1
fi

echo "Job $SLURM_JOB_ID  array $SLURM_ARRAY_TASK_ID  task=$TASK  $(date)"
echo "Node: $(hostname)  Start: $(date)"
echo "=========================================="

cd /home/aparnabg/orcd/scratch/all_project_files/vjepa_feature_extraction

python clips_vjepa.py \
    --splits_csv  /home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv \
    --out_root    /orcd/data/satra/002/projects/SAILS/vjepa_features/clips_no_chunking \
    --task        $TASK \
    --batch_clips 2 \
    --gpu         0 \
    --task_id     $SLURM_ARRAY_TASK_ID \
    --num_tasks   4

echo "End $(date)"