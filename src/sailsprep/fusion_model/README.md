# fusion_model

Late-fusion scripts that combine window-level predictions from multiple
trained model families (V-JEPA2 attentive probe, PySKL skeleton models,
PoseC3D) into a single sliding-window classifier, plus the PySKL
sliding-window dataset/eval pipeline that feeds those predictions.

## Layout

```
fusion_model/
  vjepa/
    vjepa_sw.py            trains a sliding-window attentive probe on pre-extracted V-JEPA2 features
  pyskl/
    build_pyskl_sw_pkl.py  builds PySKL sliding-window .pkl datasets + configs from full videos
    eval_pyskl_sw.py       evaluates trained PySKL sliding-window checkpoints
  late_fusion/
    two_model.py           late-fuses V-JEPA + PySKL sliding-window predictions
    three_model.py         late-fuses V-JEPA + PySKL + PoseC3D sliding-window predictions
```

All five scripts read a shared split CSV (`SPLITS_CSV` constant near the top
of each file — the `train`/`val`/`test` splits and label paths) and use a
30-frame (2 s) sliding window with 15-frame (1 s) stride over each video's
annotation timeline. Output/checkpoint directories are also set as constants
per script.

## `vjepa/vjepa_sw.py`

Trains an attentive-probe classifier over pre-extracted per-frame V-JEPA2
features (read from the split CSV column
`vjpe_features_full_video_vit_h_features`), majority-voting a label per
window from the `Locomotion` or `Repetitive_Motor_Movements` annotation
column.

```bash
python -m sailsprep.fusion_model.vjepa.vjepa_sw --task locomotion --seed 42
python -m sailsprep.fusion_model.vjepa.vjepa_sw --task rmm --seed 42
```

`--task` (`locomotion` or `rmm`) and `--seed` are both required. Writes to
`<output_base>/<task>/seed_<seed>/`.

## `pyskl/build_pyskl_sw_pkl.py` and `pyskl/eval_pyskl_sw.py`

Build the sliding-window pose dataset PySKL trains on, and evaluate trained
PySKL checkpoints against it. These operate on the same PySKL workspace used
by `action_model_testing/pyskl/` (a separate PySKL checkout with the configs
copied in) but build a different, sliding-window-labeled `.pkl` file rather
than the clip-level one used there.

```bash
# Build the .pkl dataset + generate PySKL configs for a task
python -m sailsprep.fusion_model.pyskl.build_pyskl_sw_pkl --task locomotion
python -m sailsprep.fusion_model.pyskl.build_pyskl_sw_pkl --task rmm

# Regenerate configs only, without rebuilding the .pkl (if it already exists)
python -m sailsprep.fusion_model.pyskl.build_pyskl_sw_pkl --task locomotion --configs_only

# Evaluate trained sliding-window PySKL checkpoints (aggregates 3 seeds)
python -m sailsprep.fusion_model.pyskl.eval_pyskl_sw --task locomotion
python -m sailsprep.fusion_model.pyskl.eval_pyskl_sw --task rmm
```

`--task` accepts `locomotion`, `rmm`, or `both`.

Corresponding SLURM scripts: `jobs/fusion_model/pyskl/train_pyskl_sw.sh`
(array job over 3 seeds, takes `MODEL` and `TASK` as positional arguments,
e.g. `sbatch train_pyskl_sw.sh posec3d locomotion`) and
`jobs/fusion_model/pyskl/test_pyskl_sw.sh`.

## `late_fusion/two_model.py` and `late_fusion/three_model.py`

Late-fuse per-window softmax scores from independently trained models using a
weighted average, sweeping over fusion weights to find the best combination.

`two_model.py` fuses the V-JEPA sliding-window probe with the best PySKL
sliding-window model per task (ST-GCN++ bone stream for locomotion, CTR-GCN
joint-motion stream for RMM):

```bash
python -m sailsprep.fusion_model.late_fusion.two_model \
  --task locomotion --seed 42 --alphas 0.2 0.3 0.4 0.5 0.6
```

`three_model.py` adds a PoseC3D sliding-window model as a third prediction
source, sweeping both a V-JEPA/PySKL weight (`--alphas`) and a PoseC3D weight
(`--betas`):

```bash
python -m sailsprep.fusion_model.late_fusion.three_model \
  --task rmm --seed 42 --alphas 0.2 0.3 0.4 0.5 0.6 --betas 0.1 0.2 0.3 0.4 0.5
```

Both take `--task {locomotion,rmm,both}` (default `both`) and `--seed`
(default: runs all three seeds `42, 123, 456` if omitted).
