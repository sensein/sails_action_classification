# internvideo2

Fine-tunes InternVideo2-Stage2_6B (`OpenGVLab/InternVideo2-Stage2_6B`, loaded
via Hugging Face `transformers`) for the `loco` and `rmm` classification
tasks. Only the vision encoder is used; the multimodal text components are
deleted after loading to save memory.

## File

```
internvideo2_finetune.py   single-file pipeline: data prep, dataset, model, PyTorch Lightning training + inference
```

## Data

Reads a master split CSV (`SPLIT_CSV` constant near the top of the file) with
columns `video_path`, `label_path`, `interpolated_anno_h5`, `split`. Frame-level
annotation CSVs (`label_path`) supply the `Locomotion` or
`Repetitive_Motor_Movements` column, and contiguous action runs are chunked
into fixed-length clips (`CLIP_FRAMES = 30` annotation frames, minimum
`MIN_FRAMES = 15`). Each clip is cropped to the bounding box in
`interpolated_anno_h5` on the fly at load time.

## Usage

Only the last transformer block, `LayerNorm`, and the classification head are
unfrozen by default. The model is built lazily inside Lightning's `setup()`
(after the CUDA device is set) to avoid a `cudaGetDeviceCount` crash under
`torchrun` with multiple processes.

```bash
# Single GPU, default seed (42)
torchrun --standalone --nproc_per_node=1 \
  internvideo2_finetune.py --task loco

# Specific seed
torchrun --standalone --nproc_per_node=1 \
  internvideo2_finetune.py --task loco --seed 123 --gpus 1
```

Flags:
- `--task {loco,rmm}` (required)
- `--seed <int>` (default: 42)
- `--gpus <int>` (default: 1) — number of GPUs per node passed to `--nproc_per_node`

Each seed writes to its own `seed_<N>/` subdirectory under the task's
configured output directory: model checkpoints
(`iv2-<task>-s<seed>-{epoch}-{val_loss}.ckpt`), `test_predictions.csv`, and
`test_metrics.txt`.

SLURM job: `jobs/action_model_testing/internvideo2/internvideo2_finetune.sh`,
which submits a 3-task array (seeds 42/123/456) for a given task:

```bash
sbatch jobs/action_model_testing/internvideo2/internvideo2_finetune.sh loco
sbatch jobs/action_model_testing/internvideo2/internvideo2_finetune.sh rmm
```

## Notes

- The Hugging Face cache directories (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`,
  `TRANSFORMERS_CACHE`) are redirected at the top of the file before
  `transformers` is imported — update those paths for your environment.
- InternVideo2-Stage2_6B is a large download (~12 GB) on first run.
- Training uses `bf16-mixed` precision and gradient checkpointing on the
  vision encoder.
