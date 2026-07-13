# pyskl

Skeleton-based action recognition using [PySKL](https://github.com/kennymckormick/pyskl).
This folder does **not** contain the PySKL codebase itself — it's a set of
custom scripts (pose-to-pkl conversion, config generation, logit extraction,
and a fusion MLP) that must be copied into an actual PySKL checkout to run.

## Layout

```
pyskl/
  scripts/
    pyskl_dataset.py             convert ViTPose per-clip JSONs + split CSV into a PySKL .pkl annotation file
    generate_stgcnpp_configs.py  generate ST-GCN++ configs (joint/bone/joint-motion/bone-motion)
    generate_ctrgcn_configs.py   generate CTR-GCN configs (joint/bone/joint-motion/bone-motion)
    generate_posec3d_configs.py  generate PoseC3D configs
  fusion/
    extract_logits.py            run a trained PySKL checkpoint and dump per-clip logits
    train_mlp_fusion.py          train an MLP over concatenated logits from multiple trained models
```

The corresponding SLURM scripts are at `jobs/action_model_testing/pyskl/`:
`pyskl_job.sh` (submits training across models and seeds), `test_job.sh`
(tests trained checkpoints), and `fusion_job.sh` (logit extraction + fusion).

## Setup

1. Clone and install the upstream PySKL repo (see its README for full
   environment setup — mmcv, torch, etc.):

   ```bash
   git clone https://github.com/kennymckormick/pyskl.git
   cd pyskl
   pip install -e .
   ```

2. Copy this folder's contents into the root of that clone, alongside the
   matching SLURM scripts:

   ```bash
   cp -r scripts fusion /path/to/pyskl/
   cp -r /path/to/jobs/action_model_testing/pyskl /path/to/pyskl/jobs
   ```

   (`tools/dist_train.sh`, `tools/dist_test.sh`, and the `pyskl` Python
   package used by `extract_logits.py` all come from the upstream clone —
   they are not included here.)

## Pipeline / run order

All commands below are run **from the root of the upstream PySKL clone**
(`cd /path/to/pyskl` first), after copying these files in — not from inside
`scripts/` or `fusion/`. The scripts read/write paths like
`configs/custom/...` and `work_dirs/...` relative to your current directory,
not relative to the script's own location.

1. **Convert raw pose JSONs to PySKL's `.pkl` annotation format:**

   ```bash
   python scripts/pyskl_dataset.py --task rmm  --out /path/to/rmm_pyskl.pkl
   python scripts/pyskl_dataset.py --task loco --out /path/to/loco_pyskl.pkl
   ```

2. **Generate model configs** (writes into `configs/custom/<model>_<dataset>/`):

   ```bash
   python scripts/generate_stgcnpp_configs.py
   python scripts/generate_ctrgcn_configs.py
   python scripts/generate_posec3d_configs.py
   ```

3. **Train** all model/feature combinations across 3 seeds via SLURM:

   ```bash
   bash jobs/pyskl_job.sh
   ```

4. **Test** the best checkpoint from each run:

   ```bash
   bash jobs/test_job.sh
   ```

5. **Extract per-model logits** on val + test splits, for fusion:

   ```bash
   python fusion/extract_logits.py --dataset rmm  --split val
   python fusion/extract_logits.py --dataset rmm  --split test
   python fusion/extract_logits.py --dataset loco --split val
   python fusion/extract_logits.py --dataset loco --split test
   ```

6. **Train the fusion MLP** over the concatenated logits and evaluate:

   ```bash
   python fusion/train_mlp_fusion.py --dataset rmm
   python fusion/train_mlp_fusion.py --dataset loco
   ```

## Note: not the same pipeline as `fusion_model/pyskl/`

`src/sailsprep/fusion_model/pyskl/` (`build_pyskl_sw_pkl.py`,
`eval_pyskl_sw.py`) is a separate, sliding-window PySKL pipeline used for late
fusion with other model families, with its own SLURM scripts under
`jobs/fusion_model/pyskl/`. This folder's scripts build a clip-level dataset
and train the underlying PySKL models directly; the `fusion_model/pyskl/`
scripts consume trained PySKL sliding-window models as part of the broader
fusion pipeline described in the top-level repo README.
