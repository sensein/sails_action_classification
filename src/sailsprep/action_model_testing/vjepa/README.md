# vjepa

V-JEPA2-based action recognition pipelines: feature extraction, attentive-probe
training, and head ablations, across four data setups (clip-level, full-video,
bbox-cropped clips) for two label sets (locomotion, RMM).

## Folder structure

```
vjepa/
├── common/                          Code shared across multiple top-level folders
│   ├── probes.py                    LinearProbe, MLPSmallProbe, MLPLargeProbe
│   └── bbox_utils.py                load_bbox_map
│
├── clips_without_coi_crop/          Clip-level pipeline (no bbox cropping)
│   ├── common/                      Code shared between locomotion/ and rmm/
│   │   ├── extraction.py            extract_all_features
│   │   ├── probes.py                AttentiveProbe
│   │   └── datasets.py              FeatureDataset
│   ├── locomotion/                  Locomotion
│   │   ├── extract_features.py      One-time feature extraction (frozen ViT-G)
│   │   ├── train_probe.py           Attentive probe training, per seed
│   │   └── train_probe_ablation.py  Head ablation training, per seed/head
│   └── rmm/                         RMM 
│       ├── extract_features.py
│       ├── train_probe.py
│       └── train_probe_ablation.py
│
├── clips_fixed_length/
│   └── vjepa_clip_level_ablation.py Clip-level classification with head ablation
│                                     (linear, mlp_small, mlp_large, attentive, transformer)
│
├── coi_crop/
│   └── finetune_vjepa2_h5bbox.py    End-to-end fine-tuning on H5-bbox-cropped action segments
│
└── full_video/
    ├── window.py                    Sliding-window attentive probe on full-video features
    ├── train_probe_framelevel.py    Per-frame classification, flat attentive probe
    ├── train_probe_framelevel_hierarchical.py  Per-frame classification, hierarchical (None vs action) probe
    ├── two_stage.py                 Sliding-window hierarchical (None vs action) probe
    └── rerun_window_inference.py    Re-runs window/two-stage inference at per-frame granularity
```

## Modules

### `common/`
Shared across more than one top-level folder.

- **`probes.py`** — `LinearProbe`, `MLPSmallProbe`, `MLPLargeProbe`. Used by
  `clips_without_coi_crop/locomotion/train_probe_ablation.py` and
  `clips_fixed_length/vjepa_clip_level_ablation.py`.
- **`bbox_utils.py`** — `load_bbox_map`, reads an interpolated bounding-box H5
  file into a `{frame_index: (x1, y1, x2, y2)}` map. Used by
  `clips_fixed_length/vjepa_clip_level_ablation.py` and
  `coi_crop/finetune_vjepa2_h5bbox.py`.

### `clips_without_coi_crop/`
Clip-level classification without bbox cropping, split by label set
(`locomotion/`, `rmm/`), with code shared between the two in
`clips_without_coi_crop/common/`.

- **`common/extraction.py`** — `build_dataset_from_folders` (scans a
  class-subfolder directory tree for `.mp4` clips), `VJEPA2VideoDataset`
  (uniform frame sampling via decord + VJEPA2 processor), `extract_all_features`
  (runs the frozen encoder over all clips, returns stacked features/labels).
- **`common/probes.py`** — `AttentiveProbe`: single learned query,
  cross-attention over patch tokens, linear classifier.
- **`common/datasets.py`** — `FeatureDataset`: wraps pre-extracted
  `(features, labels)` tensors for `DataLoader`.
- **`locomotion/`** / **`rmm/`** — each contains:
  - `extract_features.py` — one-time run, saves `extracted_features.pt` and
    `dataset_meta.json` shared by all seeds.
  - `train_probe.py` — loads shared features, does a seeded train/test split,
    trains `AttentiveProbe`, saves predictions and metrics per seed.
  - `train_probe_ablation.py` — same, but selectable head
    (`linear` / `mlp_small` / `mlp_large` / `attentive` / `transformer`) per run.

### `clips_fixed_length/`
- **`vjepa_clip_level_ablation.py`** — end-to-end clip-level pipeline: finds
  action runs in per-frame annotation CSVs, chunks them into fixed-length
  clips, crops by bbox, extracts VJEPA2 features, and trains/evaluates all
  five probe heads (or one, via `--head`).

### `coi_crop/`
- **`finetune_vjepa2_h5bbox.py`** — PyTorch Lightning fine-tuning of the VJEPA2
  encoder (frozen by default, or unfrozen with `--full_finetune`) plus an
  attentive pooling head, on H5-bbox-cropped action segments.

### `full_video/`
Operates on pre-extracted full-video features rather than per-clip features.

- **`window.py`** — slides a 2s/30-frame window (1s/15-frame stride) over each
  video, majority-votes a label per window, trains a flat `AttentiveProbe`.
- **`train_probe_framelevel.py`** — predicts one label per frame using an
  11-frame (±5) local context window, flat probe.
- **`train_probe_framelevel_hierarchical.py`** — same per-frame setup, but a
  two-stage probe (None vs. not-None, then specific action).
- **`two_stage.py`** — window-based version of the hierarchical probe.
- **`rerun_window_inference.py`** — re-runs inference for `window.py` and
  `two_stage.py` models, projecting window-level predictions down to
  per-frame predictions (majority vote over overlapping windows).

## Usage

```bash
# Clip-level, locomotion
python clips_without_coi_crop/locomotion/extract_features.py
python clips_without_coi_crop/locomotion/train_probe.py --seed 42
python clips_without_coi_crop/locomotion/train_probe_ablation.py --seed 42 --head attentive

# Clip-level, RMM
python clips_without_coi_crop/rmm/extract_features.py
python clips_without_coi_crop/rmm/train_probe.py --seed 42
python clips_without_coi_crop/rmm/train_probe_ablation.py --seed 42 --head attentive

# Fixed-length clips with head ablation
python clips_fixed_length/vjepa_clip_level_ablation.py --label loco --head all
python clips_fixed_length/vjepa_clip_level_ablation.py --label rmm --head all

# H5-bbox fine-tuning
python coi_crop/finetune_vjepa2_h5bbox.py
python coi_crop/finetune_vjepa2_h5bbox.py --full_finetune

# Full-video (windowed / per-frame)
python full_video/window.py --task locomotion --seed 42
python full_video/two_stage.py --task locomotion --seed 42
python full_video/train_probe_framelevel.py --task locomotion --seed 42
python full_video/train_probe_framelevel_hierarchical.py --task locomotion --seed 42
python full_video/rerun_window_inference.py --all
```

## Dependencies

`torch`, `numpy`, `pandas`, `scikit-learn`, `h5py`, `opencv-python`,
`decord`, `transformers`, `pytorch_lightning`.
