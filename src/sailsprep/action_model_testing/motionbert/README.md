# motionbert

MotionBERT-based pipeline for the locomotion clip classifier (`walk` /
`cruise` / `crawl` / `vehicle` / `run`): extracts 2D pose with YOLO11-Pose,
lifts it to 3D with MotionBERT, then fine-tunes a MotionBERT (`DSTformer`)
encoder + classification head on the 3D pose sequences.

## File

```
motionbert.py   single-file pipeline: 2D pose -> 3D lifting -> action recognition finetune/inference
```

## Setup

This script expects a local clone of
[MotionBERT](https://github.com/Walter0807/MotionBERT) with its pretrained
checkpoints, at the path set by `WORK_DIR`/`MOTIONBERT_ROOT` near the top of
the file (`sys.path.insert` is used to import `lib.model.DSTformer` and
`lib.utils.tools` from that clone at runtime — it is not installed as a
package).

Required checkpoints inside the MotionBERT clone:
- `checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin` (2D->3D lifting)
- `checkpoint/pretrain/MB_release/latest_epoch.bin` (pretrained encoder used
  to initialize the action-recognition model)

Both are available from the MotionBERT project's Hugging Face repo, e.g.:

```bash
cd MotionBERT
huggingface-cli download walterzhu/MotionBERT \
  checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin --local-dir .
```

2D pose extraction uses `ultralytics` YOLO11-Pose (`yolo11n-pose.pt`,
auto-downloaded on first run).

## Data

Reads a master split CSV (`MASTER_CSV` constant, or the `MASTER_CSV`
environment variable) with a clip-path column (`cut_clip_path` by default) and
a `split` column. The class name for each clip is inferred from **the clip's
parent folder name** and mapped through `CSV_CLASS_TO_INTERNAL` to the five
internal classes.

## Usage

```bash
cd motionbert
python motionbert.py --step all --device cuda
```

`--step` runs a subset of the pipeline: `pose2d`, `pose3d`, `finetune`,
`inference`, or `all` (default). `--use-2d-for-action` skips the 3D lifting
step and trains directly on 2D poses. `--class-weight` enables inverse-frequency
class weighting in the fine-tuning loss (off by default).

Paths are configured via environment variables (with hardcoded fallbacks in
the script): `OUTPUT_ROOT` (2D/3D pose cache), `ACTION_OUTPUT_ROOT`
(checkpoints/predictions/logs), `MASTER_CSV`.

SLURM job: `jobs/action_model_testing/motionbert/motionbert.sh`.

## Outputs

Under `ACTION_OUTPUT_ROOT`: `checkpoints/` (`best_action_model.pth`,
`latest_action_model.pth`, `final_action_model.pth`), `predictions/action_predictions.json`,
`logs/training_log.json`, `metadata/dataset_info.json` and `data_splits.json`.
Pose caches are written under `OUTPUT_ROOT/pose_2d/` and `OUTPUT_ROOT/pose_3d/`.

Training and inference both resume from existing checkpoints/predictions if
interrupted and re-run.
