#!/bin/bash
#SBATCH --job-name=fusion
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjepa_code_frame_window/new_experiments/logs/fusion_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjepa_code_frame_window/new_experiments/logs/fusion_%A_%a.err


export HF_HOME=/home/aparnabg/.cache/huggingface
export PYTHONUNBUFFERED=1
module load miniforge/24.3.0-0
module load cudnn
module load cuda
conda deactivate
source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2-312

SEEDS=(42 456 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Task: ${TASK}  Seed: ${SEED}  $(date)"
python /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjepa_code_frame_window/new_experiments/fusion_sw.py \
    --task locomotion --seed ${SEED}
echo "Done: $(date)"
