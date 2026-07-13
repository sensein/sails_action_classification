# feature_extraction

Per-frame feature extraction over full videos, cropped to the subject
bounding box. Produces the `.npy` feature files consumed by `mstcn2/` and the
PySKL/OpenTAD backbones elsewhere in this repo.

## Layout

```
feature_extraction/
  common/
    bbox.py               load_bbox_map, crop_frame_with_bbox — shared H5 bbox loading + cropping
  i3d_extractor.py         I3D + R(2+1)D features from torchvision video models
  vjepa2_extractor.py      V-JEPA2 (facebook/vjepa2-vitg-fpc64-256) features
```

Both extractor scripts import their shared helper with `from common.bbox
import ...`, so run them with `feature_extraction/` as the current directory
(not as a `python -m` module).

## Inputs

Both scripts take a split CSV with columns:

- `video_path` — path to the source video
- `interpolated_full_h5` — HDF5 with a `bboxes/table` dataset (per-frame
  bounding boxes), used to crop each frame to the subject before extracting
  features

## Usage

I3D + R(2+1)D (extracts both backbones from a single frame read per video,
output shape `(512, T)` each):

```bash
cd feature_extraction
python i3d_extractor.py \
  --splits_csv /path/to/split.csv \
  --output_dir /path/to/output \
  --target_fps 15 \
  --crop_size 224 \
  --batch_size 8 \
  --gpu 0
```

Saves to `<output_dir>/i3d/<basename>.npy` and
`<output_dir>/r2plus1d/<basename>.npy`. Pass `--backbone i3d` or `--backbone
r2plus1d` to run only one. `--overwrite` re-extracts files that already
exist.

V-JEPA2 (output shape `(1408, T)`):

```bash
cd feature_extraction
python vjepa2_extractor.py \
  --splits_csv /path/to/split.csv \
  --output_dir /path/to/output \
  --target_fps 15 \
  --crop_size 256 \
  --batch_clips 2 \
  --gpu 0
```

Both scripts support sharding a CSV across a SLURM job array with `--task_id`
and `--num_tasks` (row range `[task_id * ceil(N/num_tasks), ...]`).

Corresponding SLURM job scripts:
`jobs/action_model_testing/feature_extraction/i3d_r2p1d_feature_extracion.sh`
(pass `i3d` or `r2plus1d` as the first argument) and
`jobs/action_model_testing/feature_extraction/vjepa_feature_extracion.sh`.

## Notes

- The V-JEPA2 extractor downloads `facebook/vjepa2-vitg-fpc64-256` from
  Hugging Face on first run.
- Both scripts skip videos whose output `.npy` already exists unless
  `--overwrite` is passed, so a SLURM array job can be resubmitted safely.
