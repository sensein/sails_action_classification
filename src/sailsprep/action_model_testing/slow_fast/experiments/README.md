# experiments/

SlowFast fine-tuning experiments for the locomotion-clip classifier
(`walk` / `cruise` / `crawl` / `vehicle` / `run`). One entry point
(`ablation/train.py`) runs everything — a ten-version sweep that includes
the plain baseline (v1) and the class-weighted-loss comparison (v2) as two
of its versions, plus backbone-unfreezing, oversampling, LR, batch size,
crop size, and a model-architecture swap.

This folder does **not** include `slow_fast.py` (the repo root's separate,
self-contained pipeline that cuts its own clips from raw video + annotation
CSV + bbox H5 — a different input format entirely). See the top-level repo
for that script.

## Layout

```
experiments/
├── common/                    # shared code — no experiment logic of its own
│   ├── labels.py               # class-name <-> index mappings
│   ├── data.py                 # CSV split loader + batch collate fns
│   ├── pack_pathway.py          # SlowFast slow/fast pathway split transform
│   └── video_dataset.py         # video-clip Dataset (decord + retry-fallback)
└── ablation/
    └── train.py                # 10-version sweep (--version v1..v10)
```

`ablation/train.py` inserts `experiments/` onto `sys.path` at import time so
`from common... import ...` resolves regardless of your working directory or
whether you invoke it as `python train.py` or
`python -m experiments.ablation.train`.

## Shared input: the CSV

`ablation/train.py` reads a CSV of pre-cut clips (path configurable via the
`CSV` environment variable; falls back to a hardcoded cluster path if
unset):

| Column            | Meaning                                                        |
|-------------------|------------------------------------------------------------------|
| `cut_clip_path`   | Path to a pre-cut `.mp4` clip                                    |
| `split`           | One of `train` / `val` / `test` — assigned ahead of time, not recomputed |

The class name for each clip is inferred from **the clip's parent folder
name** (e.g. `.../Walking/clip123.mp4` → `Walking`), then mapped through
`common/labels.py`'s `CSV_CLASS_TO_INTERNAL` dict to the five internal class
names (`walk`, `cruise`, `crawl`, `vehicle`, `run`).

Because the split column is fixed in the CSV rather than recomputed by the
script, every version trains/validates/tests on the **same clips in the same
split** — results are directly comparable across versions. The loader also
asserts there is no path overlap between train/val/test.

## common/

- **`labels.py`** — `CSV_CLASS_TO_INTERNAL` (folder name → internal class),
  `ACTION_CLASSES` (ordered list of the 5 classes), `CLASS_TO_IDX` /
  `IDX_TO_CLASS`.
- **`data.py`** — `load_splits_from_csv(CSV, clip_col, split_col)`
  reads the  CSV, maps folder names to classes, and returns
  `(train_df, val_df, test_df)`. Also `slowfast_collate()` (batches the
  slow+fast pathway pair) and `slow_collate()` (batches a single pathway,
  used when `--version v10` swaps in the Slow-only model).
- **`pack_pathway.py`** — `PackPathway(alpha=4)`, a `torch.nn.Module`
  transform that splits a uniformly-sampled clip into SlowFast's
  `[slow, fast]` pathway pair.
- **`video_dataset.py`** — `VideoDataset`, a `torch.utils.data.Dataset` that
  decodes a clip with `decord` and applies the given transform. **On a
  decode failure it tries up to 10 other clips before falling back to a
  zero-tensor dummy**, so one corrupt/missing clip doesn't take down a whole
  epoch. Takes `clip_duration`, `num_frames`, `crop_size`, `alpha`, and
  `model_name` (`"slowfast_r50"` vs `"slow_r50"`, which changes the shape of
  the dummy fallback tensor) as explicit constructor args rather than reading
  module-level globals.

## ablation/train.py

Runs one of ten preset configurations, selected with `--version`:

| Version | What changes from baseline (v1) |
|---|---|
| v1  | Baseline: frozen backbone, no class weights, LR=1e-4 |
| v2  | + class-weighted loss |
| v3  | Unfreeze backbone, LR=1e-5 |
| v4  | + oversample minority classes (`WeightedRandomSampler`) |
| v5  | Class weights + unfreeze backbone, LR=1e-5 |
| v6  | Class weights + oversampling + unfreeze backbone, LR=1e-5 |
| v7  | Higher LR (1e-3), frozen backbone |
| v8  | Larger batch size (8), frozen backbone |
| v9  | Smaller crop size (224 instead of 256) |
| v10 | Swaps the model to Slow R50 instead of SlowFast R50 |

Class-weight computation (when enabled) is inverse-frequency, computed from
the train split only, and applied to **training loss only** — validation
loss is always unweighted regardless of version, so `val_loss`
(checkpoint-selection / early-stopping metric) reflects true unweighted
generalization instead of being skewed toward minority classes.

Freeze-block selection is **dynamic** — it freezes every block except the
last one, detected from `len(self.model.blocks)`, rather than a hardcoded
block name, so it also works when `--version v10` swaps in a
different-shaped model.

**Outputs** (under `MODEL_SAVE_DIR = .../output_ablation/{version}/`):
`label_mapping.json`, `test_split.csv`, `predictions_{version}.csv`,
`test_metrics_{version}.txt`, checkpoints named
`slowfast-{version}-{epoch:02d}-{val_loss:.3f}.ckpt`.

## Running

```bash
cd experiments/ablation
python train.py --version v1   # baseline
python train.py --version v2   # class-weighted loss
python train.py --version v5   # class weights + unfrozen backbone
```

Override the input CSV without editing code:

```bash
CSV=/path/to/my_split.csv python train.py --version v1
```



