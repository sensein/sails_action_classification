# slow_fast

SlowFast (`slowfast_r50`, via `torch.hub`) fine-tuning for the `loco` and
`rmm` classification tasks, bounding-box-cropped directly from full videos.

## Layout

```
slow_fast/
  slow_fast.py       end-to-end pipeline: builds clips from a split CSV + H5 bboxes, trains, and evaluates
  experiments/        locomotion-clip ablation sweep over pre-cut .mp4 clips — see experiments/README.md
```

`slow_fast.py` and `experiments/` take different inputs and are not
interchangeable: `slow_fast.py` cuts its own bbox-cropped clips on the fly
from full videos + an interpolated-annotation H5 file, while `experiments/`
reads a CSV of already pre-cut `.mp4` clips. See
[experiments/README.md](experiments/README.md) for the ablation sweep.

## `slow_fast.py`

Reads a split CSV (`SPLIT_CSV` constant near the top of the file) with
columns `video_path`, `label_path`, `interpolated_anno_h5`, `split`.
Contiguous action runs from the `Locomotion` or `Repetitive_Motor_Movements`
column are chunked into clips (`CLIP_FRAMES = 30` annotation frames, minimum
`MIN_FRAMES = 15`), each cropped to the subject's bounding box from
`interpolated_anno_h5` and split into SlowFast's slow/fast pathway pair.

```bash
cd slow_fast
python slow_fast.py --label loco
python slow_fast.py --label rmm
```

`--label` is the only CLI flag; batch size, learning rate, epochs, freeze
setting, and output directory are constants in the file
(`BATCH_SIZE`, `MAX_EPOCHS`, `LEARNING_RATE`, `FREEZE_BACKBONE`,
`LABEL_CONFIGS[label]["output_dir"]`).

SLURM job: `jobs/action_model_testing/slow_fast/slow_fast.sh`.
