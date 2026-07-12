# VideoMAE V2 Fine-tuning

Fine-tunes VideoMAE V2 ViT-B/16 (K710-distilled) on bbox-cropped video clips
for two classification tasks:

- **loco** — `Locomotion`
- **rmm** — `Repetitive_Motor_Movements`

Three training pipelines cover different windowing/labeling strategies, plus
a standalone class-weighted variant that drives the upstream VideoMAEv2
`run_class_finetuning.py` script directly.

## Layout

```
.
├── modeling_finetune.py              # vendored from OpenGVLab/VideoMAEv2 (model def)
├── utils/                            # code shared across the training scripts
│   ├── bbox.py                       # load_bbox_map()
│   ├── windowing.py                  # get_window_label()
│   ├── collate.py                    # collate()
│   └── checkpoint.py                 # K710 checkpoint download/load, build_videomae2_vitb()
├── videomae2_finetune.py             # per-action-run clips, single head
├── videomae2_fullvideo_sliding.py    # 2s/1s sliding windows, single head (+ N/A class)
├── videomae2_twostage_sliding.py     # sliding windows, binary (N/A vs active) + fine-grained heads
```

## Setup

```bash
pip install torch torchvision timm pytorch_lightning opencv-python h5py \
            pandas numpy scikit-learn

# Model definition (vendored above, but if you need to re-fetch it):
wget -O modeling_finetune.py \
  "https://raw.githubusercontent.com/OpenGVLab/VideoMAEv2/master/models/modeling_finetune.py"

# Distilled ViT-B K710 checkpoint (auto-downloaded on first run if missing):
mkdir -p ~/.cache/videomae2
wget -O ~/.cache/videomae2/vit_b_k710_dl_from_giant.pth \
  "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth"
```

All three top-level scripts read a split CSV (`SPLIT_CSV` constant near the
top of each file) with `video_path`, `label_path`, `split`, and an
h5-bbox-table column (`interpolated_anno_h5` or `interpolated_full_h5`
depending on the script).

## Usage

Run each script from the repo root (so `utils/` and `modeling_finetune.py`
resolve as imports):

```bash
python videomae2_finetune.py --task loco --seed 42
python videomae2_fullvideo_sliding.py --task loco --seed 42
python videomae2_twostage_sliding.py --task loco --seed 42
```

`--task` is `loco` or `rmm`; `--seed` defaults to 42. Each script writes
checkpoints, label mappings, predictions, and metrics to its own
`output_dir/seed_<seed>/` (see `TASK_CONFIG` in each file).

The class-weighted variant is configured via environment variables instead
of flags:


## Shared code

`load_bbox_map`, `get_window_label`, `collate`, and the K710
checkpoint-download/load path used by `build_videomae2_vitb` were identical
(or behaviorally identical) copy-pasted across two or more of the three
sliding/clip scripts. They now live in `utils/` and are imported rather than
redefined. Each script still owns its own dataset class, data module, model
wiring, and metrics/inference logic, since those genuinely differ between
the per-clip, single-head-sliding, and two-stage approaches.
