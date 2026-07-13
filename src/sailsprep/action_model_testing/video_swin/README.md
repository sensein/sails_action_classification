# video_swin

Video Swin Transformer (Swin-B) fine-tuning for action recognition on annotated
video clips, using an HDF5 bounding-box annotation format and Kinetics-400
pretrained weights.

## Structure

```
common/
  video_swin_transformer.py   Video Swin Transformer 3D backbone
  utils.py                    Shared helpers: load_bbox_map, collate_fn, VideoSwinClassifier
clip_based/
  video_swin_finetune.py      Fine-tunes on individual action-clip crops
sliding_window/
  video_swin_fullvideo_sliding.py   Multi-class classifier on 2-sec sliding windows over full videos (N/A included as a class)
  video_swin_binary_sliding.py      Binary classifier on 2-sec sliding windows (N/A vs non-N/A) - stage 1 of a two-stage pipeline
  video_swin_twostage_joint.py      Single model, shared backbone, joint binary + action heads
```

## Requirements

Installed via the `video_swin` Poetry group (`poetry install --with
dev,video_swin`): torch, pytorch-lightning, opencv-python-headless, h5py,
scikit-learn, einops, timm, pandas, numpy.

## Setup

The Swin-B backbone lives at `common/video_swin_transformer.py`
(credit: [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)).
On first run, each script downloads the official Kinetics-400 checkpoint
(`swin_base_patch244_window877_kinetics400_22k.pth`) and caches it to
`~/.cache/video_swin/`.

Each script also expects:
- An HDF5 file per video with a `bboxes/table` dataset holding per-frame bounding boxes.
- A split CSV with annotated action labels.

Paths (`TASK_CONFIG`, `SPLIT_CSV`, output directories, etc.) are set as
constants near the top of each script - edit those for your environment
before running.

## Tasks

| Task | Label column                | Classes |
|------|------------------------------|---------|
| loco | Locomotion                   | 5       |
| rmm  | Repetitive_Motor_Movements   | 4       |

(`sliding_window` scripts add an extra N/A class where applicable.)

## Usage

```bash
# Clip-based fine-tuning
cd clip_based
python video_swin_finetune.py --task loco
python video_swin_finetune.py --task loco --seed 123

# Full-video sliding-window (multi-class incl. N/A)
cd sliding_window
python video_swin_fullvideo_sliding.py --task loco

# Binary sliding-window (N/A vs non-N/A)
python video_swin_binary_sliding.py --task loco

# Joint two-stage (binary + action heads, shared backbone)
python video_swin_twostage_joint.py --task loco
```

Flags:
- `--task {loco,rmm}` (required)
- `--seed <int>` (default: 42) - each seed writes to its own output subdirectory

## Notes

- Training freezes all backbone layers except the last stage
  (`backbone.layers.3`) and the classification head(s) by default.
- Each run writes checkpoints, logs, and `test_predictions.csv` to the
  task's configured output directory.
