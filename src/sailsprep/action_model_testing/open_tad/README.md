# OpenTAD Locomotion / RMM

Scripts for preparing a custom locomotion dataset and training feature-based
temporal action detection (TAD) models on it using [OpenTAD](https://github.com/sming256/OpenTAD).

## Files

```
prepare_data/
  prepare_data.py         — converts a split CSV + per-video labels/features/pose
                             into OpenTAD-compatible annotations and .npy features.
run/
  run.py                   — generates OpenTAD configs and runs training/testing
                             for actionformer, tridet, and dyfadet across
                             backbones and seeds.
evaluate/
  common.py                 — shared helpers 
  evaluate.py               — class-aware mAP + Recall @ [0.3-0.7], mean ± std
                             + 95% CI across seeds; saves evaluation_results.csv/.txt.
  evaluate_localization.py — class-agnostic aMAP + Recall @ [0.1-0.7],
                             mean ± std across seeds, printed to stdout.
```

Named `prepare_data/`, `run/`, `evaluate/` rather than `tools/` because
OpenTAD already ships its own `tools/` folder (`tools/train.py`,
`tools/test.py`), which `run.py` calls into.

## Installation

1. Clone OpenTAD:

   ```bash
   git clone https://github.com/sming256/OpenTAD.git
   cd OpenTAD
   ```

2. Install OpenTAD's dependencies following its own README (mmaction2, mmcv,
   mmengine, etc.), then install the package:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Copy the `prepare_data/`, `run/`, and `evaluate/` folders into the OpenTAD
   repo root:

   ```bash
   cp -r /path/to/prepare_data /path/to/OpenTAD/
   cp -r /path/to/run          /path/to/OpenTAD/
   cp -r /path/to/evaluate     /path/to/OpenTAD/
   ```

All commands below must be run with the OpenTAD repo root as the current
working directory — the scripts create/read `configs/`, `exps/`, and
`data/` relative to `cwd`, not relative to their own file location. For
example, run `python run/run.py ...`, not `cd run && python run.py ...`.

## 1. Prepare the data

```bash
python prepare_data/prepare_data.py \
  --split_csv /path/to/latest_split_csv.csv \
  --output_dir data/locomotion \
  --task both \
  --ann_fps 15.0
```

`--task` accepts `locomotion`, `rmm`, or `both` (default `both`).

The split CSV must have columns: `video_path`, `label_path`, `split`,
`vjepa_full_path`, `i3d_full_path`, `r2plus1d_full_path`, `vitpose_full_path`.
Label CSVs must have `Frame` plus `Locomotion` and/or
`Repetitive_Motor_Movements` columns.

This writes:

```
data/locomotion/
  annotations/
    locomotion_anno.json
    locomotion_category_idx.txt
    rmm_anno.json
    rmm_category_idx.txt
    feature_dims.json
  features/
    vjepa/       *.npy, missing_files.txt
    i3d/         *.npy, missing_files.txt
    r2plus1d/    *.npy, missing_files.txt
    pose/        *.npy, missing_files.txt
```

## 2. Train and test

`run.py` supports `--task locomotion` or `--task rmm`, and models
`actionformer`, `tridet`, `dyfadet`.

Generate configs only (useful to inspect before training):

```bash
python run/run.py --task locomotion --model actionformer --mode generate_config
```

Train and test a single model/backbone/seed:

```bash
python run/run.py --task locomotion --model actionformer --backbone i3d \
  --mode train_test --seed 42 --gpus 1
```

Train and test one model across all backbones and all seeds (with
aggregation):

```bash
python run/run.py --task locomotion --model actionformer --mode train_test --gpus 1
```

Train and test all models × all backbones × all seeds:

```bash
python run/run.py --task locomotion --mode train_all --gpus 1
```

Test only, using an existing checkpoint:

```bash
python run/run.py --task locomotion --model actionformer --backbone i3d \
  --mode test --seed 42 \
  --checkpoint exps/locomotion/actionformer_i3d/seed_42/<run>/checkpoint/best.pth
```

Aggregate mAP across seeds:

```bash
python run/run.py --task locomotion --model actionformer --mode aggregate
```

Repeat any of the above with `--task rmm` for the RMM task.

## 3. Evaluate

Two evaluators are provided; both scan `exps/<task>/<model>_<backbone>/seed_*/*/result_detection.json`
for completed runs and compare against the `test` subset of
`locomotion_anno.json` / `rmm_anno.json`. Neither takes arguments — run them
any time after at least one `test`/`train_test` run has produced
`result_detection.json`.

Class-aware mAP + Recall @ tIoU [0.3, 0.4, 0.5, 0.6, 0.7], with mean ± std
and 95% CI across seeds; also saves `evaluation_results.csv` and
`evaluation_results.txt`:

```bash
python evaluate/evaluate.py
```

Class-agnostic aMAP (averaged over tIoU 0.1–0.7) + Recall @ tIoU
[0.1, 0.3, 0.5, 0.7], mean ± std across seeds, printed to stdout only:

```bash
python evaluate/evaluate_localization.py
```

## Options

| Flag | Description |
|---|---|
| `--task` | `locomotion` or `rmm` |
| `--model` | `actionformer`, `tridet`, or `dyfadet` |
| `--backbone` | `vjepa`, `i3d`, `r2plus1d`, or `pose`; omit to run all |
| `--mode` | `generate_config`, `train`, `test`, `train_test`, `train_all`, `aggregate` |
| `--seed` | single seed; omit to run `42, 123, 456` |
| `--gpus` | number of GPUs for `torchrun` |
| `--checkpoint` | checkpoint path for `test` mode |

## Output layout

```
exps/<task>/<model>_<backbone>/seed_<seed>/<run>/checkpoint/best.pth
exps/<task>/<model>_<backbone>/seed_summary.json
```
