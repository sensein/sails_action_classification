#!/bin/bash
# Run inference with train/test split

if [ $# -lt 2 ]; then
    echo "Usage: bash run.sh MODEL TEST_CSV [TRAIN_CSV] [OUTPUT_DIR]"
    echo ""
    echo "Arguments:"
    echo "  MODEL: llava-next-7b, qwen2-vl-7b, timezero-7b, smolvlm2-500m, videollama2-7b"
    echo "  TEST_CSV: Test data CSV path (processes all rows)"
    echo "  TRAIN_CSV: (Optional) Train data CSV for prompts"
    echo "  OUTPUT_DIR: (Optional) Output directory"
    echo ""
    echo "Example:"
    echo "  bash run.sh timezero-7b /orcd/scratch/Automatic_Labeling/test.csv /orcd/scratch/Automatic_Labeling/train.csv results_timezero"
    exit 1
fi

MODEL=$1
TEST_CSV=$2
TRAIN_CSV=${3:-""}
OUTPUT_DIR=${4:-"output_${MODEL}"}

MODEL_CACHE="/home/aparnabg/orcd/scratch/video_context_activity/my_models"
CONDA_PATH="/home/aparnabg/orcd/scratch/miniconda3"

export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
export HF_HUB_CACHE="$MODEL_CACHE"

source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate vlm_stable

if [ -z "$TRAIN_CSV" ]; then
    python inference.py \
        --model $MODEL \
        --test_csv "$TEST_CSV" \
        --output_dir "$OUTPUT_DIR"
else
    python inference.py \
        --model $MODEL \
        --test_csv "$TEST_CSV" \
        --train_csv "$TRAIN_CSV" \
        --output_dir "$OUTPUT_DIR"
fi

conda deactivate
