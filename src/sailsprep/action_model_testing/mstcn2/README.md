# mstcn2

MS-TCN++ (dual-dilated prediction generation + refinement stages) for
frame-level action segmentation, trained directly on pre-extracted per-frame
features (I3D, V-JEPA, or R(2+1)D — see `feature_extraction/` and
`vjepa/full_video/`) instead of raw video.

## File

```
mstcn2.py   model, dataset, trainer, predictor, and multi-seed aggregation in one file
```

## Data

Reads a split CSV (`SPLIT_CSV` constant near the top of the file) with
columns `video_path`, `label_path`, `split`, plus one feature-path column per
backbone (`i3d_full_path`, `vjepa_full_path`, `r2plus1d_full_path`). Feature
files can be `.npy`, `.h5`/`.hdf5`, or `.pt`, and are auto-transposed to
`(D, T)` if stored as `(T, D)`. Frame-level labels come from the `Locomotion`
or `Repetitive_Motor_Movements` column of each video's `label_path` CSV;
unlabeled/`N/A` frames map to a `background` class.

## Usage

```bash
cd mstcn2
python mstcn2.py --label loco --feature_type i3d --action train --seed 42
python mstcn2.py --label rmm  --feature_type vjepa --action train --seed 123
```

Flags:
- `--label {loco,rmm}` (required)
- `--feature_type {i3d,vjepa,r2plus1d}` (default: `i3d`)
- `--action {train,predict,evaluate}` (default: `train`) — `train` also runs
  prediction + evaluation on the val/test splits afterward; `predict` runs
  inference from an existing checkpoint; `evaluate` aggregates metrics across
  the three seeds (`42, 123, 456`)
- `--seed <int>` (default: 42)

Each seed writes to its own `seed_<N>/` subdirectory under
`<output_dir>/<feature_type>/`: `best_model.pt`, `training_history.csv`,
`test_frame_predictions.csv`, `test_segment_summary.csv`, `test_metrics.json`,
`test_report.txt`, `test_confusion_matrix.csv`. `--action evaluate` writes
`<output_dir>/<feature_type>/test_aggregate_seeds.csv` with mean/std across
seeds.

SLURM job: `jobs/action_model_testing/mstcn2/mstcn2.sh`, driven by SLURM
`--export` variables rather than positional args:

```bash
# Train all 3 seeds, then aggregate
sbatch --export=LABEL=loco,FEATURE=i3d jobs/action_model_testing/mstcn2/mstcn2.sh

# Predict only, single seed
sbatch --export=LABEL=loco,FEATURE=i3d,ACTION=predict,SEED=42 jobs/action_model_testing/mstcn2/mstcn2.sh

# Aggregate existing seed results only
sbatch --export=LABEL=loco,FEATURE=i3d,ACTION=evaluate jobs/action_model_testing/mstcn2/mstcn2.sh
```
