# Video Language Model Inference Framework

Framework for running video-language models on activity recognition with proper train/test split and weighted metrics for imbalanced classes.

## Models

Five models are supported:

- `llava-next-7b`: LLaVA-NeXT Video 7B
- `qwen2-vl-7b`: Qwen2.5-VL 7B  
- `timezero-7b`: TimeZero ActivityNet 7B (trained on ActivityNet dataset)
- `smolvlm2-500m`: SmolVLM2 500M
- `videollama2-7b`: VideoLLaMA2 7B
## ToDo
Add all Prompt versions, and models
## Installation

```bash
bash setup.sh
```

Creates conda environment `vlm_stable` with all dependencies. (for llama model need its own env)

## Usage

### With train/test split 

```bash
bash run.sh MODEL TEST_CSV TRAIN_CSV OUTPUT_DIR
```

Example:
```bash
bash run.sh timezero-7b /orcd/scratch/Automatic_Labeling/test.csv /orcd/scratch/Automatic_Labeling/train.csv results_timezero
```

This:
- Uses all 2652 samples from train.csv for prompt examples
- Evaluates on all 663 samples from test.csv

### Test model

```bash
bash run.sh MODEL TEST_CSV
```

Example:
```bash
bash run.sh timezero-7b /orcd/scratch/Automatic_Labeling/test.csv
```

### Compare models

```bash
python compare_results.py output_*/ --output comparison.csv
```

## Metrics

The framework reports both weighted and macro metrics to handle class imbalance:

- Accuracy: Overall correctness
- Weighted F1: F1 score weighted by class frequency
- Macro F1: Unweighted average F1 across classes
- BLEU: Activity description similarity


## Configuration

Edit paths in `run.sh` if needed:
- `MODEL_CACHE`: HuggingFace cache directory  
- `CONDA_PATH`: Conda installation path

## Data Format

CSV files must contain:
- `BidsProcessed`: Path to video file
- `Activity`: Activity description
- `Context`: Context label (book share, toy play, motor play, daily routine, general social communication interaction, social routine, other, special occasion)

## Output

Each run creates directory with:
- `predictions.csv`: Model predictions
- `evaluation.json`: Performance metrics (including weighted F1)


