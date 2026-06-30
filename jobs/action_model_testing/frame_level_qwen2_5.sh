#!/bin/bash
#SBATCH --job-name=qwen2_5
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --array=0-17   # 6 video chunks × 3 seeds = 18 jobs (0–17)
#SBATCH --output=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/frmae_levle/logs/qwen2_5_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/frmae_levle/logs/qwen2_5_%A_%a.err

cd /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/frmae_levle
mkdir -p logs

module load miniforge
module load cuda
module load cudnn
conda deactivate
source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
conda activate ovis_qwen_env

export HF_HOME="/orcd/data/satra/002/huggingface"
export TRANSFORMERS_CACHE="/orcd/data/satra/002/huggingface"
export HF_DATASETS_CACHE="/orcd/data/satra/002/huggingface"

CSV_FILE="/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
OUTPUT_DIR="/orcd/data/satra/002/projects/SAILS/vjepa_features/action_model_outputs/vlm_models/qwen2_5/frame_level"
SCRIPT_DIR="/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vlm_models/frmae_levle"

mkdir -p $OUTPUT_DIR

# --- Seed and chunk mapping ---
# 6 video chunks (0–5) × 3 seeds = 18 jobs
# Task IDs 0–5   → seed 123, chunks 0–5
# Task IDs 6–11  → seed 42,  chunks 0–5
# Task IDs 12–17 → seed 456, chunks 0–5

SEEDS=(123 42 456)
NUM_CHUNKS=6

SEED_IDX=$(( $SLURM_ARRAY_TASK_ID / $NUM_CHUNKS ))
CHUNK_IDX=$(( $SLURM_ARRAY_TASK_ID % $NUM_CHUNKS ))
SEED=${SEEDS[$SEED_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: seed=${SEED}, chunk=${CHUNK_IDX}"

TOTAL_VIDEOS=$(tail -n +2 $CSV_FILE | wc -l)
VIDEOS_PER_JOB=$(( ($TOTAL_VIDEOS + $NUM_CHUNKS - 1) / $NUM_CHUNKS ))
START_LINE=$(( $CHUNK_IDX * $VIDEOS_PER_JOB + 2 ))
END_LINE=$(( $START_LINE + $VIDEOS_PER_JOB - 1 ))

if [ $END_LINE -gt $(($TOTAL_VIDEOS + 1)) ]; then
    END_LINE=$(($TOTAL_VIDEOS + 1))
fi

TEMP_CSV="${OUTPUT_DIR}/temp_videos_task${SLURM_ARRAY_TASK_ID}_seed${SEED}.csv"
head -n 1 $CSV_FILE > $TEMP_CSV
sed -n "${START_LINE},${END_LINE}p" $CSV_FILE >> $TEMP_CSV

python3 ${SCRIPT_DIR}/qwen2_5.py \
    --csv $TEMP_CSV \
    --column video_path \
    --output-dir $OUTPUT_DIR \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --sample-rate 0.5 \
    --seed $SEED

rm $TEMP_CSV